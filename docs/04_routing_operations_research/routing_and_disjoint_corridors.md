# Operations Research & Dynamic Evacuation Routing

AeroSpatium converts spatial hazard forecasts into dynamic, time-dependent edge traversal costs on real road networks, computing non-overlapping evacuation corridors.

---

## 1. Dynamic Edge-Cost Function (Monotonic BPR)

Standard routing algorithms fail during natural disasters because they assume static road travel times. AeroSpatium models dynamic edge cost as a coupling of traffic congestion and physical environmental hazards.

### 1.1 Extended Bureau of Public Roads (BPR) Formulation
$$W_e(t) = T_{0, e} \cdot \left[1 + \alpha \left(\frac{V_e(t)}{C_e}\right)^\gamma\right] + \beta \cdot P_{\text{burn}, e}(t) + \delta \cdot \text{SmokePenalty}_e(t)$$

* $T_{0, e}$: Free-flow traversal time (length / posted speed limit).
* $V_e(t) / C_e$: Volume-to-capacity ratio (congestion exponent $\alpha = 0.15, \gamma = 4.0$).
* $P_{\text{burn}, e}(t)$: Spatial burn probability mapped from overlapping H3 hexagonal cells.
* $\text{SmokePenalty}_e(t)$: PM2.5 visibility and toxicity penalty ($\text{seconds} \cdot (\mu\text{g/m}^3 / 100)$).
* $\beta$: Time penalty multiplier ($3,600 \text{ s}$ per unit burn probability).

### 1.2 Mathematical Proof of Monotonicity
In emergency evacuation routing, an unconstrained neural network predicting edge costs can experience negative gradients, suggesting that an active fire or thick smoke makes a road "faster". AeroSpatium mathematically forbids this:
* Weights in the hazard projection subnet are parameterized as $W = \exp(\theta)$.
* Because $\exp(\theta) > 0$ for all real $\theta$:
  $$\frac{\partial W_e}{\partial P_{\text{burn}}} > 0 \quad \text{and} \quad \frac{\partial W_e}{\partial \text{Smoke}} > 0 \quad \forall \, \text{inputs}$$
* **Hard Edge Closure**: When $P_{\text{burn}, e}(t) \ge \tau_{\text{critical}} = 0.75$, the edge weight is set to $10^9 \text{ seconds}$, completely removing the edge from graph search feasibility.

---

## 2. Time-Dependent A* Search Algorithm

The traversal cost of road edge $e = (u, v)$ depends directly on the departure time from node $u$:
$$t_{\text{arrival}}(v) = t_{\text{departure}}(u) + W_{uv}(t_{\text{departure}}(u))$$

### 2.1 Admissible Heuristic Function
To guarantee optimal path extraction without underestimating travel times, the heuristic distance $h(u, \text{goal})$ is formulated using the Haversine great-circle distance divided by the maximum physical vehicular speed limit:

$$h(u) = \frac{d_{\text{Haversine}}(u, \text{goal})}{v_{\text{max}}}, \quad v_{\text{max}} = 30.0 \text{ m/s } (\approx 108 \text{ km/h})$$

Because no vehicle on the network can physically exceed $v_{\text{max}}$:
$$h(u) \le d^*(u, \text{goal}) \implies h(u) \text{ is strictly admissible and consistent.}$$

---

## 3. Yen's K-Shortest Paths for Edge-Disjoint Corridors

Channelling an entire evacuating population into a single "optimal" highway creates catastrophic bottleneck traps. AeroSpatium computes primary and secondary evacuation corridors that are **strictly edge-disjoint**.

```
                        (Origin: Paradise, CA)
                                  o
                                 / \
               Primary Corridor /   \ Secondary Corridor
               (Highway 70)    /     \ (Skyway Bypass)
                              o       o
                              |       |
                              o       o
                               \     /
                                \   /
                                  o
                         (Destination: Chico, CA)
```

### 3.1 Corridor Extraction Algorithm
1. **Primary Route Computation**:
   $$(P_{\text{nodes}}, P_{\text{edges}}, T_{\text{primary}}) = \text{A}^*(G, \text{start}, \text{goal}, t_0, W(t))$$
2. **Edge Penalization & Exclusion**:
   For each edge $e \in P_{\text{edges}}$ in the primary corridor, apply a punitive artificial multiplier:
   $$W_e' = W_e \times 100.0 + 10^6 \text{ seconds}$$
3. **Secondary Route Computation**:
   $$(S_{\text{nodes}}, S_{\text{edges}}, T_{\text{secondary}}) = \text{A}^*(G', \text{start}, \text{goal}, t_0, W'(t))$$
4. **Overlap Verification**:
   $$\text{OverlapFraction} = \frac{|P_{\text{edges}} \cap S_{\text{edges}}|}{|P_{\text{edges}}|} \le \text{MaxOverlapThreshold}$$
   If the secondary route shares zero critical road segments, it is issued to local emergency services as the designated fail-safe redundancy corridor.
