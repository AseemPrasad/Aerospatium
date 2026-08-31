#!/usr/bin/env python3
"""OSM PBF deep profile via pyosmium."""
import osmium as o
from collections import Counter
import json

OUT = []
def p(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s); print(s)

PATH = 'data/raw/osm/aoi.osm.pbf'

# ---- header ----
rd = o.io.Reader(PATH)
h = rd.header()
rd.close()
p("header box        :", h.box)
p("header source     :", h.get('source') if h.get('source') else None)

class H(o.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.n_nodes = 0; self.n_ways = 0; self.n_rels = 0
        self.nd_nodes = 0          # tagged nodes (POIs)
        self.unnamed_node_refs = 0
        self.highway_vals = Counter(); self.railway_vals = Counter()
        self.maxspeed_vals = Counter(); self.lanes_vals = Counter()
        self.oneway_vals = Counter(); self.surface_vals = Counter()
        self.bridge_vals = Counter(); self.access_vals = Counter()
        self.hw_ways = 0
        self.hw_missing_maxspeed = 0; self.hw_missing_lanes = 0; self.hw_missing_oneway = 0
        self.hw_missing_surface = 0; self.hw_missing_name = 0
        self.way_lens_m = []       # approx segment lengths
        self.node_lonlat_sum = [0.0, 0.0]
        self.node_count_used = 0
        self.minlon = 999; self.maxlon = -999; self.minlat = 999; self.maxlat = -999
        self.other_tags = Counter()

    def node(self, n):
        self.n_nodes += 1
        if len(n.tags) > 0:
            self.nd_nodes += 1

    def way(self, w):
        tags = {t.k: t.v for t in w.tags}
        for k in tags:
            if k not in ('highway', 'maxspeed', 'lanes', 'oneway', 'name', 'surface', 'bridge', 'access'):
                self.other_tags[k] += 1
        if 'highway' in tags:
            self.hw_ways += 1
            hw = tags['highway']
            self.highway_vals[hw] += 1
            if 'maxspeed' not in tags: self.hw_missing_maxspeed += 1
            else: self.maxspeed_vals[tags['maxspeed']] += 1
            if 'lanes' not in tags: self.hw_missing_lanes += 1
            else: self.lanes_vals[tags['lanes']] += 1
            if 'oneway' not in tags: self.hw_missing_oneway += 1
            else: self.oneway_vals[tags['oneway']] += 1
            if 'surface' not in tags: self.hw_missing_surface += 1
            else: self.surface_vals[tags['surface']] += 1
            if 'name' not in tags: self.hw_missing_name += 1
            if 'bridge' in tags: self.bridge_vals[tags['bridge']] += 1
            if 'access' in tags: self.access_vals[tags['access']] += 1
            # approximate length via node locations (locations=True)
            try:
                locs = [nd.location for nd in w.nodes]
                if all(not l.valid() for l in locs[:1]):
                    return
                import math
                R = 6371000.0
                prev = None
                for l in locs:
                    if not l.valid():
                        continue
                    lon = l.lon; lat = l.lat
                    self.minlon = min(self.minlon, lon); self.maxlon = max(self.maxlon, lon)
                    self.minlat = min(self.minlat, lat); self.maxlat = max(self.maxlat, lat)
                    if prev is not None:
                        dlat = math.radians(lat - prev[0])
                        dlon = math.radians(lon - prev[1])
                        la = math.radians(lat)
                        dp = 2 * math.asin(math.sqrt(math.sin(dlat/2)**2 + math.cos(math.radians(prev[0]))*math.cos(la)*math.sin(dlon/2)**2))
                        self.way_lens_m.append(R * dp)
                    prev = (lat, lon)
            except Exception as e:
                pass

    def relation(self, r):
        self.n_rels += 1

c = H()
c.apply_file(PATH, locations=True)

p("nodes             :", c.n_nodes)
p("ways              :", c.n_ways)
p("relations         :", c.n_rels)
p("tagged nodes      :", c.nd_nodes)
p("highway ways      :", c.hw_ways)
p("\nHIGHWAY distribution:")
for k, v in c.highway_vals.most_common():
    p(f"   {k:<16}: {v}")
p("\nMISSINGNESS on highway ways:")
p(f"   maxspeed missing: {c.hw_missing_maxspeed} ({100*c.hw_missing_maxspeed/c.hw_ways:.1f}%)")
p(f"   lanes missing   : {c.hw_missing_lanes} ({100*c.hw_missing_lanes/c.hw_ways:.1f}%)")
p(f"   oneway missing  : {c.hw_missing_oneway} ({100*c.hw_missing_oneway/c.hw_ways:.1f}%)")
p(f"   surface missing : {c.hw_missing_surface} ({100*c.hw_missing_surface/c.hw_ways:.1f}%)")
p(f"   name missing    : {c.hw_missing_name} ({100*c.hw_missing_name/c.hw_ways:.1f}%)")
p("\nmaxspeed values:", dict(c.maxspeed_vals.most_common(20)))
p("lanes values   :", dict(c.lanes_vals.most_common(10)))
p("oneway values  :", dict(c.oneway_vals.most_common()))
p("surface values :", dict(c.surface_vals.most_common(10)))
p("bridge values  :", dict(c.bridge_vals.most_common(5)))
p("access values  :", dict(c.access_vals.most_common(8)))
p("railway values :", dict(c.railway_vals.most_common(8)))
if c.way_lens_m:
    import numpy as np
    L = np.array(c.way_lens_m)
    p(f"\nsegment count: {L.size:,} | total network length: {L.sum()/1000:.1f} km")
    p(f"segment len mean={L.mean():.1f}m median={np.median(L):.1f}m p95={np.percentile(L,95):.1f}m max={L.max():.1f}m")
p(f"\ngeometry extent from ways: lon({c.minlon},{c.maxlon}) lat({c.minlat},{c.maxlat})")

with open('audit_scratch/osm_out.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(OUT))
print("\nWROTE audit_scratch/osm_out.txt")
