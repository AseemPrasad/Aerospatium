# AeroSpatium System Architecture & Technical Specifications

**AeroSpatium** is a civil defense and disaster response engine designed to predict wildfire-induced road impassability and compute dynamic, safe, edge-disjoint evacuation corridors in real time.

---

## 1. High-Level System Architecture

AeroSpatium operates as a closed-loop pipeline coupling environmental sensors, physics-informed graph neural networks, traffic flow dynamics, and graph routing algorithms.

```
                           +---------------------------+
                           |   11 Multi-Modal Feeds    |
                           | (Satellite, Weather, DEM, |
                           |  Traffic, Roads, Plumes)  |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           |  Spatial Feature Store    |
                           |    (Uber H3 Hex Grid)     |
                           +-------------+-------------+
                                         |
                         +---------------+---------------+
                         |                               |
                         v                               v
             +-----------------------+       +-----------------------+
             | Physics Engine (CA)   |       | Spatiotemporal GNN    |
             | - Rothermel (1972)    |       | - GATv2 Attention     |
             | - Albini (1976)       |       | - Temporal GRU        |
             | - Numba Multicore CPU |       | - Physics-Guided Loss |
             +-----------+-----------+       +-----------+-----------+
                         |                               |
                         +---------------+---------------+
                                         |
                                         v
                           +---------------------------+
                           | Dynamic Edge-Cost Engine  |
                           | - Learned Monotonic BPR   |
                           | - Heat/Smoke Hazard Model |
                           +-------------+-------------+
                                         |
                                         v
                           +---------------------------+
                           | Disjoint Routing Engine   |
                           | - Time-Dependent A*       |
                           | - Yen's K-Shortest Paths  |
                           +-------------+-------------+
                                         |
                         +---------------+---------------+
                         |                               |
                         v                               v
             +-----------------------+       +-----------------------+
             | FastAPI Streaming API |       | MapLibre 3D Dashboard |
             | (REST & SSE Stream)   |       | (Live Spatial Map)    |
             +-----------------------+       +-----------------------+
```

---

## 2. Core Functional Subsystems

### 2.1 Spatial Topology & Discrete Global Grid System
* **Grid Standard**: Uber H3 Discrete Global Grid System (DGGS) at resolution 8 (~460-meter edge length).
* **Graph Alignment**: Every geospatial observation (continuous rasters, discrete weather stations, polylines) is aggregated into 2,526 hexagonal cells covering the target operational theater.
* **Neighborhood Adjacency**: Each cell possesses exactly 6 equidistant immediate neighbors, removing the directional distortion inherent in 4/8-connected Cartesian raster grids.

### 2.2 Machine Learning & Physics Inversion
* **Dual-Track Modeling**:
  1. *Empirical / Physical Track*: Rothermel (1972) / Albini (1976) surface fire rate-of-spread (ROS) model accelerated with Numba JIT.
  2. *Deep Graph Track*: Spatiotemporal GNN combining Edge-Conditioned Graph Attention (`GATv2Conv` / `NativeEdgeGAT`) with Gated Recurrent Units (`GRUCell`).
* **Physics-Informed Loss**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{phys}} \mathcal{L}_{\text{flux}} + \lambda_{\text{mono}} \mathcal{L}_{\text{monotonicity}}$$
  - Penalizes predicted spatial spread gradients that violate Rothermel physical flux limits.
  - Enforces irreversible burn states: $\mathcal{L}_{\text{mono}} = \text{ReLU}(P(t) - P(t+1))$.

### 2.3 Congestion & Road Edge-Cost Function
* **Monotonic BPR Function**:
  $$W_e(t) = \text{BaseTime}_e \cdot \left(1 + \alpha \left(\frac{V}{C}\right)^\gamma\right) + \beta \cdot P_{\text{burn}}(t) + \delta \cdot \text{SmokePenalty}(t)$$
* **Mathematical Safety Guarantee**:
  - Hazard weights are constrained to be strictly positive via exponential transformations ($\exp(W)$), guaranteeing that higher fire risk or smoke concentration never decreases travel cost.
  - Hard closure threshold ($\tau_{\text{critical}} = 0.75$): When burn probability exceeds 75%, edge cost transitions to $10^9$ seconds (impassable).

### 2.4 Evacuation Routing & Corridor Extraction
* **Time-Dependent A\***:
  - Admissible heuristic using Haversine great-circle distance divided by the physical free-flow speed upper bound ($30 \text{ m/s}$).
  - Dynamically samples edge weights corresponding to the vehicle's projected arrival timestamp at each node.
* **Yen’s Disjoint Corridor Extraction**:
  - Computes primary and secondary routes with strict edge-disjoint constraints.
  - Prevents disaster-scale bottlenecks where an entire evacuating population is channeled into a single vulnerable corridor.

---

## 3. Hardware Budget & Compute Footprint

The system is explicitly engineered for deployment in resource-constrained emergency operation centers (EOCs):
* **CPU Target**: 4-core modern x86_64 CPU (e.g. AMD Ryzen 3 / Intel Core i5).
* **Memory Target**: $\le 15 \text{ GB}$ resident memory footprint under peak load.
* **GPU Dependency**: **Zero** (All neural network inference, A\* routing, and Numba simulations run on CPU).
