import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
from thread_lib import helix_solid
E="manifold"; B=trimesh.boolean

CLEAR  = 1.0     # cage たわみ代
END_CLEAR = 7.0  # cage 先端と殻の天面・底面の内面の間隔＝大きい卵でリブの先が逃げられる量（板ばねのストローク）。
                 # cage はリムで殻に固定されるので、先端は普段は浮いていて天面・底面には当たらない
BUMP_N, BUMP_H, BUMP_W = 3, 0.6, 4.0  # 蓋の内側の出っ張り（リムの V 溝にはまり上の cage を蓋側に残す）: 数 / 内径からの張り出し / 幅
WALL   = 1.2     # 殻の肉厚
SKWALL = 1.6     # 蓋スカートの肉厚
MAXSLOPE = 1.0   # dR/dz 上限 = 45 度（オーバーハング禁止）
PITCH, TDEPTH, TCLR, CRESTZ = 5.0, 1.0, 0.30, 0.6
NECK_H, THR_Z0, THR_TURNS, GRV_TURNS = 9.5, 1.3, 1.1, 2.1
SEAM_GAP = 0.5   # 締め切った時の蓋スカート下端と bottom の肩の隙間（シールは首の頂面で取る）
END_D = 24.0     # 天面・底面の径（先端が逃げる分だけ空洞の先が太いので、22 では肉厚が足りない）

def pts(f):                          # 上下2枚分の (r, z)
    u=trimesh.load(f); V,F=trimesh.remesh.subdivide_to_size(u.vertices,u.faces,max_edge=0.4)
    r=np.hypot(V[:,0],V[:,1]); return np.r_[r,r], np.r_[V[:,2],-V[:,2]]
body=pts("out/cage_unit.stl")
HALF=body[1].max()
RIM_R=body[0].max(); RIM_H=body[1][body[0]>RIM_R-0.01].max()   # リムの外径・厚さ（z=0 が2枚のリムの合わせ面）
ZC = HALF+END_CLEAR      # 空洞の上下端
ZO = ZC+WALL             # 殻の上下端

zs=np.arange(-ZO-2, ZO+2, 0.1)
def env(p,off):
    ri,zi=p; d2=off*off-(zs[:,None]-zi[None,:])**2
    return np.where(d2>0, ri[None,:]+np.sqrt(np.maximum(d2,0)),0).max(1)
def cone45(R):                       # 45度コーンで膨張 -> |dR/dz|<=1 を保証、かつ R 以上
    return np.array([ (R - MAXSLOPE*np.abs(zs-z)).max() for z in zs ])
# 空洞＝ドーム（リムより先）の包絡面＋たわみ代。先端が END_CLEAR だけ逃げたときの形（リムからの高さに比例して伸びる）も含める。
# リムの部分は蓋の下穴（BORE_R）が受ける
rz=np.round(body[1][np.abs(body[1])>RIM_H+1e-3]/0.05).astype(int); rr=body[0][np.abs(body[1])>RIM_H+1e-3]
prof=np.full(rz.max()-rz.min()+1,-1.0); np.maximum.at(prof,rz-rz.min(),rr); kz=prof>=0
pz=(np.arange(len(prof))+rz.min())[kz]*0.05; pr=prof[kz]                                 # 断面の最大半径 (z, r)
st=np.linspace(0,1,21)[:,None]*END_CLEAR*np.sign(pz)*np.clip((np.abs(pz)-RIM_H)/(HALF-RIM_H),0,1)
Rbody= np.where(np.abs(zs)<=ZC, cone45(env((np.broadcast_to(pr,st.shape).ravel(),(pz+st).ravel()),CLEAR)), 0)   # 天面・底面 |z|=ZC で切る
# 外形は空洞プロファイルの法線オフセット（肉厚を 45 度領域でも保つ）
d2 = WALL*WALL-(zs[:,None]-zs[None,:])**2
Rout = np.where(d2>0, Rbody[None,:]+np.sqrt(np.maximum(d2,0)), 0).max(1)

# 首は合わせ面の下。頂面 z=-RIM_H に下の cage のリムが載り、蓋の肩 z=+RIM_H が上の cage のリムを押さえる（2枚のリムを挟む）。
# 卵のばね力はリム→首の頂面・蓋の肩で受ける。首の外径＝リムの外径なので、リムは首の頂面の全幅に載る
ZN = -RIM_H                    # 首の頂面
O  = ZN-NECK_H                 # 蓋スカートの下端（旧座標の z=0）。ねじは旧座標から O だけ下げる
ZBAND = -O+SEAM_GAP            # bottom の肩 z=-ZBAND。|z|<=ZBAND が半径 SKIRT_R の円筒の帯
NECK_R = RIM_R
MAJ_R  = NECK_R+TDEPTH
BORE_R = NECK_R+TCLR
SKIRT_R= MAJ_R+TCLR+SKWALL

# ---------- 外形（上下共通） ----------
# 外形：閉じた状態で z=0（リムの合わせ面）について上下対称（合わせ目の隙間 SEAM_GAP を除く）。
# ねじ・リムは |z|<=ZBAND の半径 SKIRT_R の円筒の帯に入る（合わせ目は帯の下端近く z=-ZBAND..O）。
# 帯の端からの距離 d の関数 f(d) を top は z=ZBAND+d、bottom は z=-ZBAND-d に置く（縦接線なので段は出ない）。
# f: 半径 SKIRT_R の超楕円 R=SKIRT_R*(1-t^n)^(1/n) → 45 度の接線円錐 → 端面 φEND_D
# 指数 n は上下両方の必要形状を包む最小値に二分法で決める（楕円の高さ b は端面径から決まる）。
g=TCLR; HG=CRESTZ/2+g+TDEPTH+2*g
GRV=np.array([(NECK_R-g,-HG),(MAJ_R+g,-CRESTZ/2-g),(MAJ_R+g,CRESTZ/2+g),(NECK_R-g,HG)])
P=[(BORE_R,z) for z in np.arange(O,RIM_H+1e-9,0.05)]                        # めねじの下穴・リムの入る穴の壁（スカート下端 O から）
for c in O+np.arange(THR_Z0-PITCH, THR_Z0-PITCH+GRV_TURNS*PITCH+1e-9, 0.05):  # めねじ溝が通る範囲
    P+=[p0+(p1-p0)*t+[0,c] for p0,p1 in zip(GRV[:-1],GRV[1:]) for t in np.linspace(0,1,20)]
P=np.array(P); d2=WALL*WALL-(zs[:,None]-P[None,:,1])**2
Rthr=np.where(d2>0, P[None,:,0]+np.sqrt(np.maximum(d2,0)), 0).max(1)           # 溝・下穴＋WALL の包絡
Rthr[zs<O]=0                                                                   # 蓋はスカート下端 O より下にはない
ZB=-ZO                                                                         # bottom の底面
REND=END_D/2
def fit(z0,sg,need):                            # z0 から sg 向きに端 ZO まで。need: 包むべき半径
    DL=ZO-abs(z0); dd=np.linspace(0,DL,int(round(DL/0.1))+1); Rneed=np.interp(z0+sg*dd,zs,need)
    def egg(b,n):
        t=np.clip(dd/b,0,1); R=SKIRT_R*(1-t**n)**(1/n)
        w=np.where((-np.diff(R)/np.diff(dd)>=MAXSLOPE)&(R[:-1]>0))[0]           # 45 度の接点（その手前の区間はすべて 45 度未満）
        return R if len(w)==0 else np.where(dd>dd[w[0]], R[w[0]]-(dd-dd[w[0]])*MAXSLOPE, R)
    def fit_b(n):                               # 端面が φEND_D になる b
        lo,hi=0.5,300.0
        for _ in range(60):
            b=(lo+hi)/2; lo,hi=(lo,b) if egg(b,n)[-1]>REND else (b,hi)
        return b
    lo,hi=2.0,30.0
    for _ in range(40):
        n=(lo+hi)/2; lo,hi=(lo,n) if (egg(fit_b(n),n)-Rneed).min()>=0 else (n,hi)
    b=fit_b(hi); R=egg(b,hi)
    print(f"egg n{hi:.2f} b{b:.2f} | margin {(R-Rneed).min():.2f} | end dia {R[-1]*2:.1f}"
          f" | max |dR/dz| {np.abs(np.diff(R)/np.diff(dd)).max():.2f}")
    assert (R-Rneed).min()>=-1e-6 and abs(R[-1]-REND)<0.05, "外形が必要形状を包めない／端面径が合わない"
    return dd,R
mir=lambda R: np.interp(-zs,zs,R)                                          # z=0 についての鏡像
dt,Rt=fit(ZBAND,1,np.maximum.reduce([Rout,mir(Rout),Rthr,mir(Rthr)]))
Regg=np.where(np.abs(zs)>ZBAND, np.interp(np.abs(zs)-ZBAND,dt,Rt), SKIRT_R)
assert (Rthr<=SKIRT_R).all() and (Rout<=SKIRT_R)[np.abs(zs)<=ZBAND].all(), "帯がめねじ溝・空洞を包んでいない"
k=(zs>=ZB)&(zs<=ZO)
NB=Rbody[(zs>=O-SEAM_GAP)&(zs<=ZN)].max()       # 首の中の空洞の最大半径
SH=Rbody[(zs>=RIM_H)&(zs<=RIM_H+1)].max()        # 蓋の肩のすぐ上の空洞の半径
print(f"   rim R{RIM_R:.2f} H{RIM_H:.1f} | neck wall {NECK_R-NB:.2f} (=rim on neck top) | rim on lid shoulder {RIM_R-SH:.2f} mm"
      f" | tip stroke {END_CLEAR} mm | bump interference {BUMP_H-TCLR:.2f} mm")
assert NECK_R-NB>=WALL and RIM_R-SH>=WALL, "リムの掛かり（首の肉厚）が足りない"
print(f"=> OD {Regg[k].max()*2:.1f} mm, H {ZO-ZB:.1f} mm, base dia {np.interp(ZB,zs,Regg)*2:.1f} / top dia {np.interp(ZO,zs,Regg)*2:.1f} mm")
def rev(R,sec=256):
    k=(R>1e-6)&(zs>=ZB-0.001)&(zs<=ZO+0.001); z,r=zs[k],R[k]
    return trimesh.creation.revolve(np.column_stack([np.r_[0.,r,0.],np.r_[z[0],z,z[-1]]]),sections=sec)
def cyl(r,z0,z1,s=256):
    c=trimesh.creation.cylinder(radius=r,height=z1-z0,sections=s); c.apply_translation([0,0,(z0+z1)/2]); return c
HS=lambda a,b: cyl(300,a,b,8)
clean=lambda m: B.intersection([m,trimesh.creation.box(extents=[500,500,500])],engine=E)

cav=B.intersection([rev(Rbody),HS(-ZC,ZC)],engine=E)
out=B.intersection([rev(Regg),HS(ZB,ZO)],engine=E)

# ---------- bottom ----------
bowl=B.union([B.intersection([out,HS(-300,O-SEAM_GAP)],engine=E), cyl(NECK_R,O-SEAM_GAP-0.1,ZN)],engine=E)
bowl=B.difference([bowl,cav],engine=E)
HT=TDEPTH+CRESTZ/2
thr=helix_solid([(NECK_R-0.6,-HT),(NECK_R,-HT),(MAJ_R,-CRESTZ/2),
                 (MAJ_R,CRESTZ/2),(NECK_R,HT),(NECK_R-0.6,HT)],
                PITCH,THR_TURNS,O+THR_Z0,runout_lo=0.2,runout_hi=0.2)
bowl=B.union([bowl,B.intersection([B.difference([thr,cav],engine=E),cyl(99,O,ZN)],engine=E)],engine=E)
bowl=clean(B.intersection([bowl,HS(ZB,300)],engine=E)); bowl.export("out/shell_bottom.stl")

# ---------- top (printed upside down) ----------
cap=B.intersection([out,HS(O,ZO)],engine=E)
cap=B.difference([cap,cav,cyl(BORE_R,O-1,RIM_H)],engine=E)
grv=helix_solid([(NECK_R-1.2,-HG),(NECK_R-g,-HG),(MAJ_R+g,-CRESTZ/2-g),
                 (MAJ_R+g,CRESTZ/2+g),(NECK_R-g,HG),(NECK_R-1.2,HG)],
                PITCH, GRV_TURNS, O+THR_Z0-PITCH)
print(f"   thread P{PITCH} crest_h{2*HT:.1f} groove_h{2*HG:.1f} land{PITCH-2*HG:.1f}mm engage{THR_TURNS*360:.0f}deg")
assert O+THR_Z0-PITCH+GRV_TURNS*PITCH+HG<ZN, "めねじ溝がリムの入る穴にかかる"
# 出っ張り：上下45°の三角断面。先端は上の cage のリムの V 溝の中で浮く（BUMP_H-TCLR だけリムに食い込む形なので、はめるときリムがたわむ）
zg=RIM_H/2; e=0.3                                              # e: 根元を殻の肉に埋める量
bump=trimesh.creation.revolve(np.array([(BORE_R+e,zg-BUMP_H-e),(BORE_R+e,zg+BUMP_H+e),(BORE_R-BUMP_H,zg),(BORE_R+e,zg-BUMP_H-e)]),sections=512)
a=BUMP_W/2/BORE_R
bumps=[]
for i in range(BUMP_N):
    t=2*np.pi*i/BUMP_N+np.array([-a,a])
    w=[(0,0,z) for z in (-5,5)]+[(40*np.cos(u),40*np.sin(u),z) for u in t for z in (-5,5)]
    bumps.append(B.intersection([bump,trimesh.convex.convex_hull(np.array(w))],engine=E))
cap=clean(B.union([B.difference([cap,grv],engine=E)]+bumps,engine=E)); cap.export("out/shell_top.stl")

for n,m in [("bottom",bowl),("top",cap)]:
    print(f"{n}: wt={m.is_watertight} bodies={m.body_count} vol={m.volume/1000:.2f}cm3 ext={np.round(m.extents,2)}")
