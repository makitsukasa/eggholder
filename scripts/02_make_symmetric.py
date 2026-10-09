import trimesh, numpy as np, os
E="manifold"; B=trimesh.boolean
m=trimesh.load("out/work/cage_quarter.stl"); m.merge_vertices(); m.fix_normals()
c=np.array([26.3,26.3,0.0]); m.apply_translation(-m.bounds[0])
r=m.copy(); r.apply_translation(-c)
r.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[0,0,1])); r.apply_translation(c)
s=B.union([m,r],engine=E)
s.apply_translation([-c[0],-c[1],-s.bounds[0][2]])
s=B.intersection([s,trimesh.creation.box(extents=[400,400,400])],engine=E)
s.export("out/work/cage_unit_sym.stl")
b=trimesh.load("out/work/cage_unit_sym.stl"); b.merge_vertices(); b.fix_normals()
print("wt",b.is_watertight,"vol?",b.is_volume,"euler",b.euler_number,"vol",round(b.volume/1000,3),"ext",np.round(b.extents,2),"zmin",round(b.bounds[0][2],4))
r2=b.copy(); r2.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[0,0,1]))
print("point-sym", round(B.intersection([b,r2],engine=E).volume/b.volume*100,3),"%")
