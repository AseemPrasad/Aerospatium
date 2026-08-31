"""OSM road-network ingestion and H3 spatial risk fusion.

Converts raw OSM .osm.pbf files into a routable directed graph,
then attaches H3 burn-probability and PM2.5 hazard attributes.
"""
from __future__ import annotations

import importlib.util
import math

import numpy as np
import pandas as pd

HAS_GEOPANDAS = importlib.util.find_spec("geopandas") is not None
HAS_OSMIUM = importlib.util.find_spec("osmium") is not None
HAS_H3 = importlib.util.find_spec("h3") is not None
HAS_STRREE = importlib.util.find_spec("shapely") is not None

if HAS_GEOPANDAS:
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.geometry import STRtree as _STRtree

# Highway type → default speed (km/h) and lanes
_DEFAULT_SPEEDS_KMH = {
    "motorway": 110.0,
    "trunk": 90.0,
    "primary": 70.0,
    "secondary": 60.0,
    "tertiary": 50.0,
    "unclassified": 45.0,
    "residential": 30.0,
    "living_street": 20.0,
    "motorway_link": 60.0,
    "primary_link": 50.0,
    "secondary_link": 40.0,
    "tertiary_link": 35.0,
}

_DEFAULT_LANES = {
    "motorway": 2,
    "trunk": 2,
    "primary": 2,
    "secondary": 2,
    "tertiary": 1,
    "unclassified": 1,
    "residential": 1,
    "living_street": 1,
    "motorway_link": 1,
    "primary_link": 1,
    "secondary_link": 1,
    "tertiary_link": 1,
}

_ROUTABLE_HIGHWAYS = set(_DEFAULT_SPEEDS_KMH.keys())


class OSMNetworkHandler:
    """Pyosmium handler that extracts routable road edges from an OSM PBF."""

    def __init__(self):
        self.nodes = {}
        self.ways = []

    def node(self, n):
        self.nodes[n.id] = (n.location.lon, n.location.lat)

    def way(self, w):
        tags = {t.k: t.v for t in w.tags}
        hw = tags.get("highway")
        if hw not in _ROUTABLE_HIGHWAYS:
            return

        access = tags.get("access", "")
        if access in ("private", "no"):
            return

        ref = [n.ref for n in w.nodes]

        # Speed
        speed_kmh = _DEFAULT_SPEEDS_KMH.get(hw, 30.0)
        maxspeed = tags.get("maxspeed")
        if maxspeed:
            speed_kmh = self._parse_maxspeed(maxspeed, speed_kmh)

        lanes = _DEFAULT_LANES.get(hw, 1)
        if "lanes" in tags:
            try:
                lanes = int(tags["lanes"])
            except ValueError:
                pass

        oneway = tags.get("oneway", "no")
        name = tags.get("name", "")

        self.ways.append({
            "osm_way_id": w.id,
            "highway": hw,
            "name": name,
            "speed_kmh": speed_kmh,
            "lanes": lanes,
            "oneway": oneway,
            "node_refs": ref,
            "capacity_veh_hr": lanes * 1900,  # veh/hr per lane default
        })

    @staticmethod
    def _parse_maxspeed(val, default):
        if val == "none" or val == "":
            return default
        num = ""
        for ch in val:
            if ch.isdigit() or ch == ".":
                num += ch
            else:
                break
        if num:
            speed = float(num)
            if "mph" in val.lower():
                speed *= 1.60934
            return speed
        return default


def ingest_osm_network(
    pbf_path: str,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest OSM PBF into road nodes and edges GeoDataFrames.

    Parameters
    ----------
    pbf_path : str
        Path to the .osm.pbf file.
    bbox : (min_lon, min_lat, max_lon, max_lat), optional
        Bounding box to crop. If None, the PBF header box is used.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Columns: node_id, osmid, lon, lat
    edges_gdf : GeoDataFrame
        Columns: u, v, edge_id, name, highway_type, speed_kmh,
                 lanes, length_m, capacity_veh_hr, base_time_s, geometry
    """
    if not HAS_OSMIUM:
        raise ImportError("pyosmium is required for OSM ingestion")
    if not HAS_GEOPANDAS:
        raise ImportError("geopandas is required")

    handler = OSMNetworkHandler()
    handler.apply_file(pbf_path, locations=True, idx="flex_mem")

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        handler.nodes = {
            k: v for k, v in handler.nodes.items()
            if min_lon <= v[0] <= max_lon and min_lat <= v[1] <= max_lat
        }

    # Build nodes GeoDataFrame
    node_records = []
    for nid, (lon, lat) in handler.nodes.items():
        node_records.append({
            "node_id": nid,
            "osmid": nid,
            "lon": lon,
            "lat": lat,
        })
    nodes_gdf = gpd.GeoDataFrame(
        node_records,
        geometry=[Point(r["lon"], r["lat"]) for r in node_records],
        crs="EPSG:4326",
    ) if node_records else gpd.GeoDataFrame(
        columns=["node_id", "osmid", "lon", "lat", "geometry"], crs="EPSG:4326"
    )

    # Build edges GeoDataFrame
    edge_records = []
    edge_id = 0
    for way in handler.ways:
        refs = way["node_refs"]
        if len(refs) < 2:
            continue

        coords = []
        valid = True
        for ref in refs:
            if ref not in handler.nodes:
                valid = False
                break
            coords.append(handler.nodes[ref])

        if not valid or len(coords) < 2:
            continue

        # Determine directionality
        fwd_speed = way["speed_kmh"]
        rev_speed = fwd_speed if way["oneway"] in ("no", "reversible", "") else 0.0

        # Create edge segments
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            # Project to UTM for accurate length
            p1 = gpd.GeoSeries([Point(lon1, lat1)], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
            p2 = gpd.GeoSeries([Point(lon2, lat2)], crs="EPSG:4326").to_crs("EPSG:5070").iloc[0]
            length_m = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)

            if length_m <= 0:
                continue

            base_time_s = length_m / (fwd_speed * 1000.0 / 3600.0)
            capacity = way["capacity_veh_hr"]

            edge_records.append({
                "u": refs[i],
                "v": refs[i + 1],
                "edge_id": edge_id,
                "name": way["name"],
                "highway_type": way["highway"],
                "speed_kmh": fwd_speed,
                "lanes": way["lanes"],
                "length_m": length_m,
                "capacity_veh_hr": capacity,
                "base_time_s": base_time_s,
                "oneway": way["oneway"],
            })
            edge_id += 1

            if rev_speed > 0 and way["oneway"] in ("no", "reversible", ""):
                rev_time_s = length_m / (rev_speed * 1000.0 / 3600.0)
                edge_records.append({
                    "u": refs[i + 1],
                    "v": refs[i],
                    "edge_id": edge_id,
                    "name": way["name"],
                    "highway_type": way["highway"],
                    "speed_kmh": rev_speed,
                    "lanes": way["lanes"],
                    "length_m": length_m,
                    "capacity_veh_hr": capacity,
                    "base_time_s": rev_time_s,
                    "oneway": way["oneway"],
                })
                edge_id += 1

    edges_gdf = gpd.GeoDataFrame(
        edge_records,
        geometry=[
            LineString([
                (handler.nodes[r["u"]], handler.nodes[r["v"]])
            ]) for r in edge_records
        ],
        crs="EPSG:4326",
    ) if edge_records else gpd.GeoDataFrame(
        columns=["u", "v", "edge_id", "name", "highway_type", "speed_kmh",
                 "lanes", "length_m", "capacity_veh_hr", "base_time_s",
                 "oneway", "geometry"],
        crs="EPSG:4326",
    )

    return nodes_gdf, edges_gdf


def fuse_h3_risks_to_edges(
    edges_df: pd.DataFrame,
    h3_static_df: pd.DataFrame,
    h3_hazard_df: pd.DataFrame,
    hazard_column: str = "burn_prob_12h",
    pm25_column: str = "pm25_baseline_ugm3",
) -> pd.DataFrame:
    """Attach H3 burn-probability and PM2.5 values to road edges.

    For each edge, finds the nearest H3 centroid and joins the
    corresponding hazard values.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Road edges with node coordinates.
    h3_static_df : pd.DataFrame
        H3 static features including centroid_lat, centroid_lon.
    h3_hazard_df : pd.DataFrame
        H3 hazard predictions with burn_prob columns.
    hazard_column : str
        Column name for the burn probability to attach.
    pm25_column : str
        Column name for PM2.5 value.

    Returns
    -------
    pd.DataFrame
        Edges with added hazard columns.
    """
    merged = edges_df.copy()

    # Build a merged H3 lookup
    h3_lookup = h3_static_df.merge(
        h3_hazard_df[["h3_id", hazard_column]],
        on="h3_id",
        how="left",
    )

    h3_points = np.column_stack([
        h3_lookup["centroid_lat"].values,
        h3_lookup["centroid_lon"].values,
    ])
    h3_values_burn = h3_lookup[hazard_column].values
    h3_values_pm25 = h3_lookup[pm25_column].values if pm25_column in h3_lookup.columns else np.zeros(len(h3_points))

    if HAS_STRREE:
        tree = _STRtree([Point(lo, la) for la, lo in h3_points])
    else:  # pragma: no cover
        tree = None

    for idx, edge in merged.iterrows():
        edge_lat = (edge.get("lat_u", 0.0) + edge.get("lat_v", 0.0)) / 2.0
        edge_lon = (edge.get("lon_u", 0.0) + edge.get("lon_v", 0.0)) / 2.0

        if tree is not None:
            pt = Point(edge_lon, edge_lat)
            nearest_idx = tree.nearest(pt)
            merged.at[idx, "h3_primary_id"] = h3_lookup.iloc[nearest_idx]["h3_id"]
            merged.at[idx, "gnn_burn_prob"] = h3_values_burn[nearest_idx]
            merged.at[idx, "pm25_ugm3"] = h3_values_pm25[nearest_idx]
        else:
            # Fallback: brute-force nearest
            dists = np.sqrt(
                (h3_points[:, 0] - edge_lat) ** 2
                + (h3_points[:, 1] - edge_lon) ** 2
            )
            nearest = np.argmin(dists)
            merged.at[idx, "h3_primary_id"] = h3_lookup.iloc[nearest]["h3_id"]
            merged.at[idx, "gnn_burn_prob"] = h3_values_burn[nearest]
            merged.at[idx, "pm25_ugm3"] = h3_values_pm25[nearest]

    return merged


def build_rustworkx_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> object:
    """Convert pandas edge tables into a rustworkx.PyDiGraph.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        Must contain node_id.
    edges_df : pd.DataFrame
        Must contain u, v, edge_id.

    Returns
    -------
    rustworkx.PyDiGraph (if available) or tuple (nodes, edges)
    """
    try:
        import rustworkx
        graph = rustworkx.PyDiGraph()
        node_map = {}
        for _, row in nodes_df.iterrows():
            nid = int(row["node_id"])
            node_map[nid] = graph.add_node({"node_id": nid, **row.to_dict()})

        for _, row in edges_df.iterrows():
            u = int(row["u"])
            v = int(row["v"])
            if u not in node_map or v not in node_map:
                continue
            graph.add_edge(node_map[u], node_map[v], {"edge_id": int(row["edge_id"]), **row.to_dict()})
        return graph
    except ImportError:  # pragma: no cover
        return nodes_df, edges_df
