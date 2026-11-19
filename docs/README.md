# AeroSpatium Documentation Library

Welcome to the comprehensive technical documentation for **AeroSpatium** — the multi-modal wildfire progression and evacuation route optimization engine.

---

## Documentation Volumes

1. [Volume 1: System Overview & Architecture](01_system_overview/architecture.md)
   - Closed-loop system topology
   - Spatial DGGS (Uber H3) grid foundations
   - Operational modes & hardware execution targets ($\le 15 \text{ GB}$ RAM, 4 CPU cores)

2. [Volume 2: Multi-Modal Data Ingestion](02_data_pipeline/ingestion_catalog.md)
   - Catalog of the 11 environmental and satellite feeds
   - Topographic processing (Horn's 3x3 DEM finite difference gradients)
   - Inverse Distance Weighting (IDW) and Anderson 13 fuel classification

3. [Volume 3: Physics-Informed Machine Learning](03_machine_learning_physics/spatiotemporal_gnn.md)
   - Rothermel (1972) & Albini (1976) combustion physics
   - Spatiotemporal Graph Neural Network (`GATv2Conv` / `NativeEdgeGAT` + `GRUCell`)
   - Physics-informed loss formulation (flux constraints and irreversible monotonicity)

4. [Volume 4: Operations Research & Evacuation Routing](04_routing_operations_research/routing_and_disjoint_corridors.md)
   - Extended Bureau of Public Roads (BPR) congestion functions
   - Monotonic neural network layers (strictly non-negative hazard penalties)
   - Time-dependent A* heuristic proofs & Yen's edge-disjoint corridor algorithm

5. [Volume 5: API Reference & Production Operations](05_engineering_operations/api_reference.md)
   - FastAPI endpoints (`/route`, `/routes/disjoint`, `/stream/fire-spread`)
   - Production deployment runbook, pytest validation, and Airflow orchestration
