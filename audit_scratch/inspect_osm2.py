#!/usr/bin/env python3
"""OSM PBF - complete handler."""
import osmium as o
from collections import Counter,defaultdict

class H(o.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.n_nodes=0; self.n_ways=0; self.n_rels=0
        self.node_tags=0; self.way_tags=0; self.rel_tags=0
        self.highway=Counter(); self.lit=Counter()
        self.maxspeed=Counter(); self.lanes=Counter(); self.oneway=Counter()
        self.highway_ways=0; self.name_ways=0; self.ref_ways=0
        self.way_tag_count=Counter()  # count tags across ways
        self.highway_sample=[]
        self.railway=Counter(); self.amenity=Counter(); self.natural=Counter()
        self.bounds=None
        self.maxspeed_vals=Counter()
        self.lanes_vals=Counter()
        self.oneway_vals=Counter()
        self.highway_vals=Counter()

    def node(self,n):
        self.n_nodes+=1
        self.node_tags+=len(n.tags)
    def way(self,w):
        self.n_ways+=1
        self.way_tags+=len(w.tags)
        tw = {}
        for tag in w.tags:
            tw[tag.k]=tag.v
            self.way_tag_count[tag.k]+=1
        if 'highway' in tw:
            self.highway_ways+=1
            self.highway_vals[tw['highway']]+=1
            if len(self.highway_sample)<5:
                self.highway_sample.append(tw)
        if 'railway' in tw: self.railway[tw['railway']]+=1
        if 'amenity' in tw: self.amenity[tw['amenity']]+=1
        if 'natural' in tw: self.natural[tw['natural']]+=1
        if 'name' in tw: self.name_ways+=1
        if 'ref' in tw: self.ref_ways+=1
        if 'maxspeed' in tw:
            self.maxspeed_vals[tw['maxspeed']]+=1
        if 'lanes' in tw:
            self.lanes_vals[str(tw['lanes'])]+=1
        if 'oneway' in tw:
            self.oneway_vals[tw['oneway']]+=1
        if 'lit' in tw: self.lit[tw['lit']]+=1
    def relation(self,r):
        self.n_rels+=1
        self.rel_tags+=len(r.tags)

c=H()
c.apply_file('data/raw/osm/aoi.osm.pbf', locations=True)

print("=== OBJECT COUNTS ===")
print("nodes :",c.n_nodes)
print("ways  :",c.n_ways)
print("rels  :",c.n_rels)
print("\n=== TAG TOTALS ===")
print("node_tags :",c.node_tags)
print("way_tags  :",c.way_tags)
print("rel_tags  :",c.rel_tags)

print("\n=== ALL TAGS ON WAYS (count) ===")
for k,v in c.way_tag_count.most_common():
    print(f"   {k:<18}: {v}")

print("\n=== HIGHWAY category distribution (ways with highway tag) ===")
for k,v in c.highway_vals.most_common():
    print(f"   {k:<20}: {v}")
print("total ways w/ highway:", sum(c.highway_vals.values()))
print("ways with name:", c.name_ways, " refs:", c.ref_ways)

print("\n=== maxspeed values ===")
for k,v in c.maxspeed_vals.most_common(): print(f"   {k:<12}: {v}")
print("=== lanes values ===")
for k,v in c.lanes_vals.most_common(): print(f"   {k:<8}: {v}")
print("=== oneway values ===")
for k,v in c.oneway_vals.most_common(): print(f"   {k:<8}: {v}")
print("\n=== highway-tagged way samples (first 5) ===")
for s in c.highway_sample: print("  ",s)

print("\n=== other features ===")
print("railway:",dict(c.railway))
print("amenity:",dict(c.amenity))
print("natural:",dict(c.natural))
print("lit:",dict(c.lit))
