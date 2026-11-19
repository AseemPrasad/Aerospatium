# Physics-Informed Machine Learning & Hazard Engine

AeroSpatium couples classical combustion physics with modern deep geometric learning to predict wildfire progression across discrete hexagonal topologies.

---

## 1. Classical Combustion Physics (Rothermel & Albini)

The rate-of-spread (ROS) model calculates the forward propagation speed of a surface fire front:

$$R = \frac{I_R \cdot \xi}{\rho_b \cdot \epsilon \cdot Q_{ig}} \cdot \left(1 + \Phi_w + \Phi_s\right)$$

### 1.1 Reaction Intensity ($I_R$)
$$I_R = \Gamma \cdot w_0 \cdot h \cdot \eta_m \cdot \eta_s \quad [\text{kJ} / \text{m}^2\text{-min}]$$
* $\Gamma$: Optimum reaction velocity based on packing ratio $\beta / \beta_{op}$.
* $w_0$: Net fuel loading ($\text{kg/m}^2$).
* $h$: Heat content ($18,600 \text{ kJ/kg}$).
* $\eta_m$: Moisture damping coefficient:
  $$\eta_m = 1 - 2.59 \left(\frac{M_f}{M_x}\right) + 5.11 \left(\frac{M_f}{M_x}\right)^2 - 3.52 \left(\frac{M_f}{M_x}\right)^3$$
* $\eta_s$: Mineral damping coefficient ($0.174 \cdot \text{Se}^{-0.19}$).

### 1.2 Directional Wind Amplification ($\Phi_w$)
Midflame wind speed $U$ is projected along the neighbor cell azimuth $\theta_{\text{neighbor}}$:
$$U_{\text{eff}} = \max\left(0, U \cdot \cos(\theta_{\text{wind}} - \theta_{\text{neighbor}})\right)$$
$$\Phi_w = C \cdot (U_{\text{eff}})^B \cdot \left(\frac{\beta}{\beta_{op}}\right)^{-E}$$
Where constants $C, B, E$ derive from fuel surface-area-to-volume ratio $\sigma$. To enforce physical validity under extreme conditions, Albini's upper bound is applied: $U_{\text{eff}} \le 0.9 \cdot I_R$.

### 1.3 Directional Topographic Slope Factor ($\Phi_s$)
$$\Phi_s = 5.275 \cdot \beta^{-0.3} \cdot \left(\max(0, \tan(\text{Slope}) \cdot \cos(\text{Aspect} - \theta_{\text{neighbor}}))\right)^2$$

---

## 2. Spatiotemporal Graph Neural Network Architecture

The deep learning model operates over an undirected graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where vertices $\mathcal{V}$ are H3 hexagons and edges $\mathcal{E}$ represent spatial adjacency.

### 2.1 Feature Representation
* **Static Node Tensors** ($X_{\text{static}} \in \mathbb{R}^{N \times 9}$): Elevation, slope, aspect, fuel model one-hot encoding, population density, historical burn frequency.
* **Dynamic Node Sequences** ($X_{\text{dyn}} \in \mathbb{R}^{T \times N \times 6}$): $u_{10}, v_{10}, T_{2m}, RH$, surface drought index, PM2.5 smoke concentration.
* **Edge Attributes** ($E_{\text{attr}} \in \mathbb{R}^{E \times 3}$): Centroid-to-centroid distance, elevation delta ($\Delta z$), edge azimuth angle.

### 2.2 Forward Computation & Attention Message Passing
For each timestep $t \in [1, \dots, T]$:
1. **Node Embedding Concatenation**:
   $$x_i^{(t)} = [X_{\text{static}, i} \,\|\, X_{\text{dyn}, i}^{(t)}] \in \mathbb{R}^{15}$$
2. **Edge-Conditioned Attention (`GATv2Conv`)**:
   $$\alpha_{ij}^{(t)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W_{\text{src}} x_i^{(t)} + W_{\text{dst}} x_j^{(t)} + W_e e_{ij}]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [W_{\text{src}} x_i^{(t)} + W_{\text{dst}} x_k^{(t)} + W_e e_{ik}]\right)\right)}$$
   $$g_i^{(t)} = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(t)} (W_{\text{src}} x_j^{(t)} + W_e e_{ij})\right)$$
3. **Temporal Recurrence (`GRUCell`)**:
   $$h_i^{(t)} = \text{GRUCell}(g_i^{(t)}, h_i^{(t-1)})$$
4. **Hazard Projection Head**:
   $$P_{\text{burn}, i}(t) = \text{Sigmoid}(W_2 \cdot \text{SiLU}(W_1 h_i^{(t)} + b_1) + b_2)$$

---

## 3. Physics-Guided Loss Formulation

Training is driven by a hybrid loss function that enforces physical laws directly into the neural network's gradient descent:

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda_{\text{phys}} \mathcal{L}_{\text{phys}} + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}}$$

```
                        +-----------------------------------------+
                        |           Total Training Loss           |
                        +----+------------------+------------+----+
                             |                  |            |
                             v                  v            v
                   +------------------+  +-----------+  +-----------+
                   | Supervised BCE   |  | Physical  |  | Temporal  |
                   | (Satellite Mask) |  | Flux PDE  |  | Monotonic |
                   +------------------+  +-----------+  +-----------+
```

1. **Supervised Loss ($\mathcal{L}_{\text{sup}}$)**: Masked Binary Cross-Entropy against sparse satellite ground truth observations (NASA FIRMS):
   $$\mathcal{L}_{\text{sup}} = \frac{\sum_{t, i} M_{t, i} \cdot \text{BCE}(P_i(t), Y_{t, i})}{\sum_{t, i} M_{t, i} + \epsilon}$$
2. **Physical Flux Constraint ($\mathcal{L}_{\text{phys}}$)**: Penalizes spatial burn probability differences across neighboring cells $(i, j)$ that exceed the maximum theoretical Rothermel rate-of-spread:
   $$\mathcal{L}_{\text{phys}} = \frac{1}{|\mathcal{E}|} \sum_{(i, j) \in \mathcal{E}} \text{ReLU}\left(|P_i(t) - P_j(t)| - R_{\text{Rothermel}, i}\right)$$
3. **Irreversible Monotonicity ($\mathcal{L}_{\text{mono}}$)**: Imposes the physical reality that burned fuel cannot spontaneously unburn within an evacuation horizon:
   $$\mathcal{L}_{\text{mono}} = \frac{1}{T \cdot N} \sum_{t=1}^{T-1} \sum_{i=1}^N \text{ReLU}\left(P_i(t) - P_i(t+1)\right)$$
