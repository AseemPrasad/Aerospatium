#!/usr/bin/env python3
"""OSM PBF inspection via pyosmid/osmium reader."""
import osmium as o
from collections import Counter

banner = lambda t: print("\n"+"="*78+"\n"+t+"\n"+"="*78)

banner("OSM PBF: data/raw/osm/aoi.osm.pbf")

# File-level header
fh = o.FileReader('data/raw/osm/aoi.osm.pbf', 'osm.pbf')
fh.close()
print("Format handler readable.")

# Header info (bounds, version)
bh = o.HeaderReader('data/raw/osm/aoi.osm.pbf', o.CompressionFormat.DETECTED).read()
print("header bounds:", bh.bounds) if bh.bounds else print("header bounds: None")
print("header version:", bh.version)
print("header generator:", bh.writingprogram)
print("header timestamp:", bh.timestamp)

# Walk handlers
class CounterHandler(o.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.nodes = 0
        self.ways = 0
        self.rels = 0
        self.node_tags = 0
        self.way_tags = 0
        self.rel_tags = 0
        self.highway = Counter()
        self.highway_in_ways = 0
        self.ways_with_highway = 0
        self.maxspeed_count = 0
        self.lanes_count = 0
        self.oneway_count = 0
        self.highway_count = 0
        self.lit_count = 0
        self.ref_count = 0
        self.name_count = 0

c = CounterHandler()
c.apply_file('data/raw/osm/aoi.osm.pbf', locations=True)

print("\n--- OBJECT COUNTS ---")
print("nodes  :", c.nodes)
print("ways   :", c.ways)
print("rels   :", c.rels)

print("\n--- TAG / FIELD MISSINGNESS (on ways, the road graph) ---")
print("ways_total_tagged (way_tags) :", c.way_tags)
print("ways with 'highway' tag      :", c.highway_count)
print("ways with 'maxspeed' tag     :", c.maxspeed_count)
print("ways with 'lanes' tag        :", c.lanes_count)
print("ways with 'oneway' tag       :", c.oneway_count)
print("ways with 'name' tag         :", c.name_count)
print("ways with 'ref' tag          :", c.ref_count)
print("ways with 'lit' tag          :", c.lit_count)
print("node_tags total              :", c.node_tags)
print("rel_tags total               :", c.rel_tags)

print("\n--- HIGHWAY CATEGORY DISTRIBUTION (all ways) ---")
for k,v in c.highway.most_common():
    print(f"   {k:<24} : {v}")
