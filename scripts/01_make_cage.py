import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
E="manifold"; B=trimesh.boolean

# 内側ケージ（上下共通・点対称）。リム下面が z=0、+z 側がドーム。
H        = 30.8    # 全高
DOME_A   = 32.1    # ドーム外面の楕円 r = A*sqrt(1-((z-ZC)/B)^2)
DOME_B   = 72.0
DOME_ZC  = -38.3
THICK    = 2.0     # ドームの肉厚（法線方向）
FLANGE_R, FLANGE_H = 27.5, 2.0   # リム（つば）
N_RIB    = 6       # 板ばねリブの本数
TWIST    = 58.0    # リブの根元→先端のねじれ角 [度]
W_BOT, W_TOP = 8.8, 5.8          # リブ幅（肉厚中央での水平方向の弧長）根元 / 先端
GAP_Z0   = FLANGE_H+0.6          # リブ間の隙間の下端（半円で丸める）
MAXSLOPE = 1.0                   # ドームの dR/dz 上限＝鉛直から45°
# 保持脚：リムから立てた脚の先端の爪を殻の環状溝に掛け、蓋を開けても cage が殻側に残るようにする。
# リブには触れない（リブ間の隙間に置く）ので板ばねの剛性は変わらない。
N_LEG    = 2                     # 本数（180°回転対称を保つため偶数）
LEG_T    = 1.2                   # 脚の肉厚。内径はリブ外面＋LEG_GAP（半径方向にもリブと離す）
LEG_GAP  = 0.3
LEG_W    = 5.0                   # 脚の幅（外径での弧長）
BARB_Z   = 10.0                  # 爪の下端（ねじ首の頂面 z=9.5 より上）
BARB_H, BARB_F = 0.6, 0.3        # 爪の張り出し（上下とも45°） / 先端の平らな部分
LEG_TOP  = BARB_Z+2*BARB_H+BARB_F

zs=np.linspace(0,H,309)
# 外面：楕円。傾きが 45 度を超える先端側は接線円錐に置き換える（cage はリム下で印刷するのでオーバーハング禁止）
zf=np.linspace(-5,H+5,4000); Rf=DOME_A*np.sqrt(np.clip(1-((zf-DOME_ZC)/DOME_B)**2,0,None))
Rf=(Rf[None,:]-MAXSLOPE*np.abs(zf[:,None]-zf[None,:])).max(1)
Ro=np.interp(zs,zf,Rf)
# 内面＝外面の法線オフセット（同じ z での水平半径を数値で求める）
d2=THICK**2-(zs[:,None]-zf[None,:])**2
Ri=np.where(d2>0, Rf[None,:]-np.sqrt(np.maximum(d2,0)), 1e9).min(1)
Rm=(Ro+Ri)/2

# リブの中心角と半幅（角度）。根元では隙間を半円で閉じてリムにつなぐ。
th_c=np.radians(TWIST)*zs/H
W=W_BOT-(W_BOT-W_TOP)*(zs/H)**2
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

v=np.linspace(-1,1,25)
ribs=[]
for n in range(N_RIB):
    th=th_c[:,None]+2*np.pi*n/N_RIB+hw[:,None]*v[None,:]
    P=np.stack([np.stack([R[:,None]*np.cos(th),R[:,None]*np.sin(th),np.broadcast_to(zs[:,None],th.shape)],-1)
                for R in (Ro,Ri)],2)
    ribs.append(grid_solid(P))
fl=trimesh.creation.annulus(r_min=np.interp(FLANGE_H,zs,Ri),r_max=FLANGE_R,height=FLANGE_H,sections=512)
fl.apply_translation([0,0,FLANGE_H/2])

# 保持脚。リムより上のリブ外面より外に置き、さらにリブ間の隙間が開いている高さでリブが通らない角度帯の中央に置く。
LEG_RI=Ro[zs>=FLANGE_H].max()+LEG_GAP; LEG_RO=LEG_RI+LEG_T
zl=(zs<=LEG_TOP)&(gap>=2*np.pi*Rm/N_RIB-W-1e-9)
lo_edge=(th_c+hw)[zl].max(); hi_edge=(th_c-hw)[zl].min()+2*np.pi/N_RIB
LEG_ANG=(lo_edge+hi_edge)/2; leg_hw=LEG_W/2/LEG_RO
prof=[(LEG_RI,0),(LEG_RO,0),(LEG_RO,BARB_Z),(LEG_RO+BARB_H,BARB_Z+BARB_H),(LEG_RO+BARB_H,BARB_Z+BARB_H+BARB_F),
      (LEG_RO,LEG_TOP),(LEG_RI,LEG_TOP)]
ring=trimesh.creation.revolve(np.array(prof+prof[:1]),sections=512)
legs=[]
for n in range(N_LEG):
    a=LEG_ANG+2*np.pi*n/N_LEG+np.array([-leg_hw,leg_hw])           # くさびで脚の幅だけ切り出す
    w=[(0,0,z) for z in (-1,LEG_TOP+1)]+[(40*np.cos(t),40*np.sin(t),z) for t in a for z in (-1,LEG_TOP+1)]
    legs.append(B.intersection([ring,trimesh.convex.convex_hull(np.array(w))],engine=E))
print(f"leg r{LEG_RI:.2f}..{LEG_RO:.2f} barb r{LEG_RO+BARB_H:.2f} | angle {np.degrees(LEG_ANG):.1f}deg | rib-free band {np.degrees(hi_edge-lo_edge):.1f}deg,"
      f" margin each side {(hi_edge-lo_edge)/2*LEG_RI-LEG_W/2:.2f}mm @r{LEG_RI}")
body=B.union(ribs+[fl],engine=E)
k=B.union([body]+legs,engine=E)
clean=lambda m: B.intersection([m,trimesh.creation.box(extents=[500,500,500])],engine=E)
clean(k).export("out/cage_unit.stl")
os.makedirs("out/work",exist_ok=True)
clean(body).export("out/work/cage_body.stl")                  # 02 の空洞（たわみ代）用
clean(B.union(legs,engine=E)).export("out/work/cage_legs.stl")  # 02 の環状溝用

b=trimesh.load("out/cage_unit.stl"); b.merge_vertices(); b.fix_normals()
r2=b.copy(); r2.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[0,0,1]))
nz=b.face_normals[:,2]; a=b.area_faces; zc=b.triangles_center[:,2]
bad=(nz<-np.sin(np.radians(45.5)))&(zc>0.01)
print("wt",b.is_watertight,"bodies",b.body_count,"vol",round(b.volume/1000,3),"ext",np.round(b.extents,2),
      "point-sym",round(B.intersection([b,r2],engine=E).volume/b.volume*100,2),"%",
      "| overhang>45 (接地面除く)",round(a[bad].sum(),1),"mm2")
if T: render(b,T+"/cage.png",elev=25,azim=-45,size=800)
if T: render(b,T+"/cage_top.png",elev=89,azim=0,size=800)
