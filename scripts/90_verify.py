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
print("[蓋を開けたとき上の cage が蓋側に残るか (上の cage を蓋から下へずらす。出っ張りがリムの V 溝に掛かる)]")
for dz in (0.2,0.4,0.8):
    c=sym.copy(); c.apply_translation([0,0,-dz])
    print(f"   top x cage上 -{dz}mm: {B.intersection([cap,c],engine=E).volume:6.2f} mm3")
print("[卵のばね力で cage が端へ押されたとき、リムで止まるか (cage ごと端へずらす)]")
for cn,cm,sn,sm,sg in [("上",sym,"top",cap,1),("下",lo,"bottom",bowl,-1)]:
    v=[]
    for dz in (0.1,0.3):
        c=cm.copy(); c.apply_translation([0,0,sg*dz]); v.append(B.intersection([sm,c],engine=E).volume)
    print(f"   {sn} x cage{cn} 0.1 / 0.3mm: {v[0]:6.2f} / {v[1]:6.2f} mm3")
END_CLEAR=cap.bounds[1][2]-1.2-sym.bounds[1][2]   # 天面の内面と cage 先端の間隔（WALL=1.2）
H=sym.bounds[1][2]; RH=2.0
print(f"[大きい卵で先端が逃げられるか (リムは固定・リブの先を端へ s×{END_CLEAR:.1f}mm。リムからの高さに比例して伸ばす)]")
for cn,cm,sn,sm,sg in [("上",sym,"top",cap,1),("下",lo,"bottom",bowl,-1)]:
    v=[]
    for s_ in (0.5,0.9,1.0,1.1):
        c=cm.copy(); z=c.vertices[:,2]*sg
        c.vertices[:,2]+=sg*s_*END_CLEAR*np.clip((z-RH)/(H-RH),0,1); v.append(B.intersection([sm,c],engine=E).volume)
    print(f"   {sn} x cage{cn} s=0.5/0.9/1.0/1.1: "+" / ".join(f"{x:.2f}" for x in v)+" mm3")
print("[卵 (03_make_eggs) が入るか。卵を上下の cage の間で釣り合う位置に置き、リブ先端の必要な変位を求める]")
# リブは高さに比例して伸びる（先端の変位 s）とみなす。s>0 で卵が押し付けられる（ばねが効く）、s<0 は隙間
V,_=trimesh.remesh.subdivide_to_size(sym.vertices,sym.faces,max_edge=0.4)
vz=np.round(V[:,2]/0.1).astype(int); vr=np.hypot(V[:,0],V[:,1])
pin=np.full(vz.max()+1,1e9); np.minimum.at(pin,vz,vr); cz=np.arange(len(pin))*0.1; cri=pin   # cage 内面 (z, r)
def need_s(xe,re):
    def ok(s_):
        zz=cz+s_*np.clip((cz-RH)/(H-RH),0,1); k=(xe>0)&(xe<=zz[-1])
        return (re[k]<=np.interp(xe[k],zz,cri)+1e-9).all()
    a,b_=-15.0,15.0
    for _ in range(50):
        m_=(a+b_)/2; a,b_=(a,m_) if ok(m_) else (m_,b_)
    return b_
ZCAV=cap.bounds[1][2]-1.2
for en in ("small","large"):
    eg=L(f"out/egg_{en}.stl"); W_,_=trimesh.remesh.subdivide_to_size(eg.vertices,eg.faces,max_edge=0.5)
    ez=np.round(W_[:,2]/0.1).astype(int); er=np.hypot(W_[:,0],W_[:,1])
    pe=np.zeros(ez.max()-ez.min()+1); np.maximum.at(pe,ez-ez.min(),er); xe=(np.arange(len(pe))+ez.min())*0.1
    best=None
    for c in np.linspace(-4,4,161):
        st=need_s(xe+c,pe); sb=need_s(-(xe+c),pe)
        if best is None or abs(st-sb)<abs(best[1]-best[2]): best=(c,st,sb)
    c,st,sb=best; e2=eg.copy(); e2.apply_translation([0,0,c])
    vol=B.intersection([e2,B.union([bowl,cap],engine=E)],engine=E).volume
    print(f"   {en}: 中心 z{c:+.2f} / 先端の変位 上 {st:+.2f} 下 {sb:+.2f} mm (逃げ代 {END_CLEAR:.1f}) / "
          f"卵の先と天面・底面 {ZCAV-e2.bounds[1][2]:.2f} / {e2.bounds[0][2]+ZCAV:.2f} mm / 殻との干渉 {vol:.2f} mm3")
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
