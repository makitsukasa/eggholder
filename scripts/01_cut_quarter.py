import trimesh, numpy as np
m = trimesh.load("ref_CyberCyclist/files/cage.stl")
m.apply_translation(-m.bounds[0])
e = m.extents
box = trimesh.creation.box(extents=[e[0]/2, e[1]/2, e[2]*2])
box.apply_translation([e[0]/4, e[1]/4, e[2]/2])
q = trimesh.boolean.intersection([m, box], engine="manifold")
q.apply_translation(-q.bounds[0])
print("watertight", q.is_watertight, "bodies", q.body_count, "vol", q.volume/1000, "size", q.extents)
q.export("out/work/cage_quarter.stl")
