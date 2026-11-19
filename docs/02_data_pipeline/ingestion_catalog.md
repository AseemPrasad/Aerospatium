# Multi-Modal Data Ingestion Catalog

AeroSpatium harmonizes 11 distinct geospatial and environmental feeds into a unified spatio-temporal feature store.

---

## 1. Data Source Inventory

| Feed ID | Data Source | Provider | Native Format | Resolution / Cadence | Operational Role |
|:---|:---|:---|:---|:---|:---|
| `FEED_01` | **NASA FIRMS / MODIS** | NASA LANCE | CSV / GeoTIFF | 375m–1km / 6–12h passes | Active fire detections & initial burn perimeters |
| `FEED_02` | **LANDFIRE EVT** | USGS / USFS | GeoTIFF | 30m / Multi-year static | Surface fuel loading & Anderson 13 fuel model mapping |
| `FEED_03` | **SRTM DEM** | USGS / NASA | GeoTIFF | 30m / Static topography | Elevation, slope angle, and aspect azimuth |
| `FEED_04` | **ERA5-Land** | ECMWF | NetCDF-4 | 0.1° (~9km) / Hourly | Wind speed ($u_{10}, v_{10}$), 2m temperature, relative humidity |
| `FEED_05` | **OpenStreetMap** | Geofabrik / OSM | `.osm.pbf` | Vector road network | Routable road graph (junctions, capacities, free-flow speeds) |
| `FEED_06` | **USDM** | NDMC / USDA | Shapefile | Weekly polygon | County-level drought severity index (D0–D4) |
| `FEED_07` | **Census TIGER** | US Census Bureau | GeoJSON | Decennial / Annual | Population density for evacuation volume estimation |
| `FEED_08` | **WFIGS** | NIFC | GeoJSON | Real-time incident stream | Ground-truth historical and active fire perimeters |
| `FEED_09` | **Caltrans PeMS** | Caltrans | CSV | 5-minute / Station point | Freeway & arterial traffic volumes, vehicle speeds ($V/C$) |
| `FEED_10` | **AirNow PM2.5** | EPA AirNow | CSV / REST | Hourly / Sensor point | Smoke plume particulate concentration ($\mu\text{g/m}^3$) |
| `FEED_11` | **NOAA HRRR** | NOAA NCEP | GRIB-2 / NetCDF | 3km / 1-hour forecast | Real-time operational surface wind & boundary layer forecast |

---

## 2. Ingestion & Transformation Matrix

### 2.1 Topographic Processing (`SRTM DEM`)
* **Raw Elevation**: $z(x, y)$ in meters above sea level.
* **Horn's 3x3 Finite Difference Gradients**:
  $$dz/dx = \frac{(z_{ne} + 2z_e + z_{se}) - (z_{nw} + 2z_w + z_{sw})}{8 \cdot \Delta x}$$
  $$dz/dy = \frac{(z_{sw} + 2z_s + z_{se}) - (z_{nw} + 2z_n + z_{ne})}{8 \cdot \Delta y}$$
* **Slope & Aspect**:
  $$\text{Slope} = \arctan\left(\sqrt{(dz/dx)^2 + (dz/dy)^2}\right)$$
  $$\text{Aspect} = \left(\frac{3\pi}{2} - \arctan2(dz/dy, -dz/dx)\right) \pmod{2\pi}$$

### 2.2 Atmospheric Spatial Interpolation
* Atmospheric parameters (temperature $T_{2m}$, relative humidity $RH$, wind vector components $u_{10}, v_{10}$) are distributed to H3 hexagonal centroids using **Inverse Distance Weighting (IDW)**:
  $$\hat{Z}(s_0) = \frac{\sum_{i=1}^k w_i(s_0) Z(s_i)}{\sum_{i=1}^k w_i(s_0)}, \quad w_i(s_0) = \frac{1}{d(s_0, s_i)^p}$$
  Where power parameter $p=2.0$ and neighborhood size $k=4$.

### 2.3 Fuel Classification (`LANDFIRE EVT` $\rightarrow$ Anderson 13)
* Raw Existing Vegetation Type (EVT) codes are classified into standard Anderson 13 surface fuel models:
  - `Model 1`: Short Grass (w0: 0.166 kg/m², fuel bed depth: 0.30m)
  - `Model 2`: Timber Grass (w0: 0.896 kg/m², fuel bed depth: 0.30m)
  - `Model 4`: Chaparral / Heavy Brush (w0: 1.120 kg/m², fuel bed depth: 1.83m)
  - `Model 10`: Timber Litter (w0: 1.346 kg/m², fuel bed depth: 0.30m)
  - `Model 99`: Non-burnable / Urban / Water bodies.
