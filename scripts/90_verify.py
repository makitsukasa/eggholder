import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
E="manifold"; B=trimesh.boolean; PITCH=5.0
L=lambda f:(lambda m:(m.merge_vertices(),m.fix_normals(),m)[2])(trimesh.load(f))
bowl=L("out/shell_bottom.stl"); cap=L("out/shell_top.stl"); sym=L("out/cage_unit.stl")
lo=sym.copy(); lo.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0])); lo.fix_normals()
print("[ねじ込み 0->540度]"); 
for d in np.arange(0,541,60):
    c=cap.copy(); c.apply_transform(trimesh.transformations.rotation_matrix(np.radians(d),[0,0,1])); c.apply_translation([0,0,d/360*PITCH])
    print(f"   {d:4.0f}deg: {B.intersection([bowl,c],engine=E).volume:7.2f} mm3")
print("[真上に引抜き(ねじ保持の確認)]")
for dz in (0.5,1.5,3.0):
    c=cap.copy(); c.apply_translation([0,0,dz])
    print(f"   {dz}mm: {B.intersection([bowl,c],engine=E).volume:7.1f} mm3")
print("[cage との干渉]")
for cn,cm in [("上",sym),("下",lo)]:
    for sn,sm in [("bottom",bowl),("top",cap)]:
        print(f"   {sn} x cage{cn}: {B.intersection([sm,cm],engine=E).volume:6.2f} mm3")
print("[オーバーハング (印刷姿勢)]")
for n,m,flip in [("bottom 底面下",bowl,False),("top 天面下(反転)",cap,True)]:
    x=m.copy()
    if flip: x.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0])); x.fix_normals()
    nz=x.face_normals[:,2]; a=x.area_faces; dn=nz<-1e-6
    ang=np.degrees(np.arcsin(np.clip(-nz[dn],0,1)))   # 90=水平天井, 0=垂直壁
    bad=ang>45.5
    print(f"   {n}: 下向き面{a[dn].sum():6.0f}mm2 / 45度超{a[dn][bad].sum():7.1f}mm2 ({a[dn][bad].sum()/m.area*100:.2f}%)")
box=trimesh.creation.box(extents=[300,150,300]); box.apply_translation([0,-75,0])
full=B.union([bowl,cap,sym,lo],engine=E)
if T: render(B.intersection([full,box],engine=E),T+"/v3_full.png",elev=3,azim=180,size=1000)
if T: render(B.intersection([B.union([bowl,cap],engine=E),box],engine=E),T+"/v3_asm.png",elev=3,azim=180,size=1000)
if T:
    render(bowl,T+"/v3_bowl.png",elev=20,azim=-45,size=800)
    render(cap,T+"/v3_cap.png",elev=20,azim=-45,size=800)
