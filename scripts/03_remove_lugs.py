import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
E="manifold"; B=trimesh.boolean
L=lambda f:(lambda m:(m.merge_vertices(),m.fix_normals(),m)[2])(trimesh.load(f))
m=L("out/work/cage_unit_sym.stl")
c=trimesh.creation.cylinder(radius=27.5,height=60,sections=512); c.apply_translation([0,0,25])
k=B.intersection([m,c],engine=E)
k=B.intersection([k,trimesh.creation.box(extents=[500,500,500])],engine=E)
k.export("out/cage_unit.stl")
b=L("out/cage_unit.stl")
r2=b.copy(); r2.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[0,0,1]))
print("wt",b.is_watertight,"bodies",b.body_count,"vol",round(b.volume/1000,3),"ext",np.round(b.extents,2),
      "point-sym",round(B.intersection([b,r2],engine=E).volume/b.volume*100,2),"%")
v=b.vertices; rr=np.hypot(v[:,0],v[:,1])
print("rmax",round(rr.max(),2))
# inner cavity (egg space): min radius of inner surface per z
for z0 in np.arange(0,31,3):
    s=(v[:,2]>=z0)&(v[:,2]<z0+3)
    if s.sum()>5: print(f"  z {z0:4.0f} r {rr[s].min():6.2f}..{rr[s].max():6.2f}")
if T: render(b,T+"/cage_nolug.png",elev=20,azim=-45,size=800)
if T: render(b,T+"/cage_nolug_top.png",elev=89,azim=0,size=800)
