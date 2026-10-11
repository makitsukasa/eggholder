import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
E="manifold"; B=trimesh.boolean

# 内側ケージ（上下共通・点対称）。リム下面が z=0、+z 側がドーム。
# ドーム内面は「半径 IN_R の円筒 → 45° の円錐（頂点 z=APEX）」。卵の両端を円錐で受け、太い卵の胴は円筒で逃がす。
# 円錐で受けると卵の中心と頂点の距離は≒√(a²+b²)（a,b: 長さ・太さの半分）。APEX は小さい卵に少し押し付けが掛かる位置にする（90_verify で確認）
IN_R     = 24.5    # 円筒部の内半径（太い卵 φ47.7 の胴を逃がす）
APEX     = 33.8    # 内面の円錐の頂点の高さ
HOLE_R   = 6.9     # 先端の穴の半径（指の先の内面）
H        = APEX-HOLE_R   # 全高
THICK    = 2.0     # ドームの肉厚（法線方向）
FLANGE_R, FLANGE_H = 29.4, 2.0   # リム（つば）。2枚重ねて bottom の首の頂面と蓋の肩で挟む。外径＝首の外径（02 はここから首を決める）
RIM_GRV  = 0.5                   # リム外周の V 溝の深さ（上下45°、リム厚の中央）。蓋の内側の出っ張りがはまり、上の cage を蓋側に残す
N_RIB    = 6       # 板ばねリブの本数
TWIST    = 58.0    # リブの根元→先端のねじれ角 [度]
W_BOT, W_TOP = 8.8, 5.8          # リブ幅（肉厚中央での水平方向の弧長）根元 / 先端
GAP_Z0   = FLANGE_H+0.6          # リブ間の隙間の下端（半円で丸める）
MAXSLOPE = 1.0                   # ドームの dR/dz 上限＝鉛直から45°
zs=np.linspace(0,H,309)
# 内面：円筒 → 45° 円錐（cage はリム下で印刷するので内面の傾きは 45° まで）
zf=np.linspace(-5,H+5,4000); Rf=np.minimum(IN_R,APEX-MAXSLOPE*zf)
Ri=np.interp(zs,zf,Rf)
# 外面＝内面の法線オフセット（同じ z での水平半径を数値で求める）
d2=THICK**2-(zs[:,None]-zf[None,:])**2
Ro=np.where(d2>0, Rf[None,:]+np.sqrt(np.maximum(d2,0)), 0).max(1)
Rm=(Ro+Ri)/2

# リブの中心角と半幅（角度）。根元では隙間を半円で閉じてリムにつなぐ。
# ねじれの進み方は、ねじれの向きの側の縁（θc+W/2Rm）の傾き r·dθ/dz がどの高さでも同じになるようにする。
# z に比例させると、円錐が続く大きい半径のところや、幅の角度が急に広がる先端でリブの側面が下向き 45° を超える
W=W_BOT-(W_BOT-W_TOP)*(zs/H)**2
hw0=W/2/Rm                                   # 隙間を丸める前の半幅 [rad]
iw=np.r_[0,np.cumsum((1/Rm[1:]+1/Rm[:-1])/2*np.diff(zs))]
k=(np.radians(TWIST)+hw0[-1]-hw0[0])/iw[-1]; th_c=k*iw-(hw0-hw0[0])
print(f"rib edge slope r*dθ/dz = {k:.3f} (side face {np.degrees(np.arctan(k)):.1f}deg from vertical)")
gap=2*np.pi*Rm/N_RIB-W                       # 隣のリブとの隙間（弧長）
g0=gap/2
t=np.clip((zs-GAP_Z0)/np.maximum(g0,1e-6),0,1)
gap=np.where(zs<GAP_Z0, 0.0, np.where(t<1, gap*np.sqrt(1-(1-t)**2), gap))
hw=(2*np.pi*Rm/N_RIB-gap)/2/Rm               # 半幅 [rad]

def grid_solid(P):
    """P[i,j,k,:]  i: z方向, j: 幅方向, k: 0=外面/1=内面 の格子を閉じた立体にする"""
    ni,nj=P.shape[:2]; V=P.reshape(-1,3); idx=np.arange(ni*nj*2).reshape(ni,nj,2); F=[]
    def quad(a,b,c,d): F.extend([[a,b,c],[a,c,d]])
    for i in range(ni-1):
        for j in range(nj-1):
            quad(idx[i,j,0],idx[i,j+1,0],idx[i+1,j+1,0],idx[i+1,j,0])        # 外面
            quad(idx[i,j,1],idx[i+1,j,1],idx[i+1,j+1,1],idx[i,j+1,1])        # 内面
        for j,s in ((0,1),(nj-1,-1)):                                         # 側面
            a,b,c,d=idx[i,j,0],idx[i+1,j,0],idx[i+1,j,1],idx[i,j,1]
            quad(a,b,c,d) if s>0 else quad(a,d,c,b)
    for i,s in ((0,1),(ni-1,-1)):                                             # 下端・上端
        for j in range(nj-1):
            a,b,c,d=idx[i,j,0],idx[i,j,1],idx[i,j+1,1],idx[i,j+1,0]
            quad(a,b,c,d) if s>0 else quad(a,d,c,b)
    m=trimesh.Trimesh(V,np.array(F),process=True); m.fix_normals(); return m

v=np.linspace(-1,1,61)
ribs=[]
for n in range(N_RIB):
    th=th_c[:,None]+2*np.pi*n/N_RIB+hw[:,None]*v[None,:]
    P=np.stack([np.stack([R[:,None]*np.cos(th),R[:,None]*np.sin(th),np.broadcast_to(zs[:,None],th.shape)],-1)
                for R in (Ro,Ri)],2)
    ribs.append(grid_solid(P))
zg=FLANGE_H/2; rin=np.interp(FLANGE_H,zs,Ri)
fl=trimesh.creation.revolve(np.array([(rin,0),(FLANGE_R,0),(FLANGE_R,zg-RIM_GRV),(FLANGE_R-RIM_GRV,zg),(FLANGE_R,zg+RIM_GRV),
                                      (FLANGE_R,FLANGE_H),(rin,FLANGE_H),(rin,0)]),sections=512)
body=B.union(ribs+[fl],engine=E)
clean=lambda m: B.intersection([m,trimesh.creation.box(extents=[500,500,500])],engine=E)
clean(body).export("out/cage_unit.stl")

b=trimesh.load("out/cage_unit.stl"); b.merge_vertices(); b.fix_normals()
r2=b.copy(); r2.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[0,0,1]))
nz=b.face_normals[:,2]; a=b.area_faces; zc=b.triangles_center[:,2]
bad=(nz<-np.sin(np.radians(45.5)))&(zc>0.01)
print("wt",b.is_watertight,"bodies",b.body_count,"vol",round(b.volume/1000,3),"ext",np.round(b.extents,2),
      "point-sym",round(B.intersection([b,r2],engine=E).volume/b.volume*100,2),"%",
      "| overhang>45 (接地面除く)",round(a[bad].sum(),1),"mm2")
if T: render(b,T+"/cage.png",elev=25,azim=-45,size=800)
if T: render(b,T+"/cage_top.png",elev=89,azim=0,size=800)
