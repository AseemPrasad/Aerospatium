#!/usr/bin/env python3
"""Vector + OSM profiling for AeroSpatium audit."""
import json, os, struct
import numpy as np

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s); print(s)
def banner(t):
    p("\n" + "=" * 78); p(t); p("=" * 78)

# =====================================================================
# DROUGHT SHAPEFILE - raw sidecar inspection
# =====================================================================
banner("DROUGHT USDM SIDECAR FILES")
base = "data/raw/drought/USDM_20260818"
for ext in [".prj", ".cpg"]:
    with open(base + ext, "r", encoding="utf-8", errors="replace") as f:
        p(f"{ext} content: {f.read()!r}")

# DBF header parse (dBase III)
with open(base + ".dbf", "rb") as f:
    hdr = f.read(32)
    version, year, month, day = hdr[0], hdr[1] + 1900, hdr[2], hdr[3]
    nrec = struct.unpack("<I", hdr[4:8])[0]
    hlen, rlen = struct.unpack("<HH", hdr[8:12])
    p(f"DBF version={version} date={year}-{month:02d}-{day:02d} records={nrec} header_len={hlen} rec_len={rlen}")
    fields = []
    pos = 32
    while True:
        f.seek(pos)
        fd = f.read(32)
        if fd[0:1] == b"\x0d":
            break
        name = fd[0:11].split(b"\x00")[0].decode()
        ftype = chr(fd[11])
        flen = fd[16]
        fdec = fd[17]
        fields.append((name, ftype, flen, fdec))
        pos += 32
    p("DBF fields (name, type, len, dec):", fields)
    # first record dump
    f.seek(hlen)
    for i in range(min(nrec, 6)):
        rec = f.read(rlen)
        off = 1
        vals = []
        for name, ftype, flen, fdec in fields:
            vals.append((name, rec[off:off + flen].decode(errors="replace")))
            off += flen
        p(f"  rec{i}:", vals)

# SHP main-file header parse
with open(base + ".shp", "rb") as f:
    sh = f.read(100)
    file_code = struct.unpack(">i", sh[0:4])[0]
    file_len_words = struct.unpack(">i", sh[24:28])[0]
    ver, shp_type = struct.unpack("<ii", sh[28:36])
    xmin, ymin, xmax, ymax = struct.unpack("<4d", sh[36:68])
    zmin, zmax, mmin, mmax = struct.unpack("<4d", sh[68:100])
    p(f"SHP file_code={file_code} length={file_len_words*2:,}B shape_type={shp_type}")
    p(f"SHP declared bbox: lon({xmin},{xmax}) lat({ymin},{ymax}) z({zmin},{zmax}) m({mmin},{mmax})")

# Geometry via duckdb spatial
import duckdb
con = duckdb.connect()
con.execute("INSTALL 'spatial'"); con.execute("LOAD 'spatial'")
p("\n--- Drought via DuckDB Spatial ---")
con.execute(f"CREATE TABLE d AS SELECT * FROM ST_READ('{base}.shp')")
p("rows:", con.execute("SELECT COUNT(*) FROM d").fetchone()[0])
p(con.execute("DESCRIBE d").fetchdf().to_string())
p(con.execute("SELECT * FROM d").fetchdf().drop(columns=["geom"]).to_string())
p("geom types:", con.execute("SELECT DISTINCT ST_GeometryType(geom) FROM d").fetchall())
p("per-feature bbox:")
for r in con.execute("""SELECT DM::INT dm, ST_XMin(geom) x0, ST_YMin(geom) y0,
                        ST_XMax(geom) x1, ST_YMax(geom) y1,
                        ROUND(ST_Area(ST_Transform(geom,'EPSG:4326','EPSG:5070'))/1e6,1) area_km2,
                        ST_NumGeometries(geom) ng
                        FROM d ORDER BY dm""").fetchall():
    p("   ", r)
p("vertices total:", con.execute("SELECT SUM(ST_NPoints(geom)) FROM d").fetchone()[0])
p("is_valid:", con.execute("SELECT DM, ST_IsValid(geom) FROM d").fetchall())

# =====================================================================
# GEOJSON: census + wfigs
# =====================================================================
from shapely.geometry import shape

def geojson_profile(name, path):
    banner(f"GEOJSON: {name} ({path}) [{os.path.getsize(path):,} B]")
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    p("top-level keys:", list(gj.keys()))
    p("type:", gj.get("type"), "| name:", gj.get("name"), "| crs member:", gj.get("crs"))
    feats = gj["features"]
    p("feature count:", len(feats))
    if feats:
        keys = list(feats[0]["properties"].keys())
        p("property keys:", keys)
        for k in keys:
            vals = [f["properties"].get(k) for f in feats]
            types = sorted({type(v).__name__ for v in vals})
            p(f"  {k}: dtype={types} sample={vals[:6]}")
        gtypes = {}
        lons, lats = [], []
        for f_ in feats:
            g = shape(f_["geometry"])
            gtypes[f_["geometry"]["type"]] = gtypes.get(f_["geometry"]["type"], 0) + 1
            b = g.bounds
            lons += [b[0], b[2]]; lats += [b[1], b[3]]
            if g.geom_type == "Polygon":
                p(f"  feat {f_['properties']}: rings={len(g.interiors)+1} interior_rings={len(g.interiors)} npts={g.exterior.coords.__len__()} bbox={tuple(round(v,6) for v in g.bounds)}")
        p("geometry types:", gtypes)
        p(f"total extent: lon({min(lons)},{max(lons)}) lat({min(lats)},{max(lats)})")
        p("bbox member:", gj.get("bbox"))

geojson_profile("Census block groups", "data/raw/census/census_blocks.geojson")
geojson_profile("WFIGS fire perimeters", "data/raw/wfigs/wfigs_perimeters.geojson")

with open("audit_scratch/vector_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\nWROTE audit_scratch/vector_out.txt")
