import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
from thread_lib import helix_solid
E="manifold"; B=trimesh.boolean

CLEAR  = 1.0     # cage たわみ代
END_CLEAR = 1.0  # cage 先端と殻の天面・底面の内面との遊び（先端の穴から卵の先が出るので、大きい卵ならここを広げる）
LEG_CLR = 0.3    # cage の保持脚・爪と環状溝の隙間
WALL   = 1.2     # 殻の肉厚
SKWALL = 1.6     # 蓋スカートの肉厚
MAXSLOPE = 1.0   # dR/dz 上限 = 45 度（オーバーハング禁止）
PITCH, TDEPTH, TCLR, CRESTZ = 5.0, 1.0, 0.30, 0.6
NECK_H, THR_Z0, THR_TURNS, GRV_TURNS = 9.5, 1.3, 1.1, 2.1
SEAM_GAP = 0.5   # 締め切った時の蓋スカート下端と bottom の肩の隙間（シールは首の頂面で取る）
END_D = 22.0     # 天面・底面の径

def pts(f):                          # 上下2枚分の (r, z)
    u=trimesh.load(f); V,F=trimesh.remesh.subdivide_to_size(u.vertices,u.faces,max_edge=0.4)
    r=np.hypot(V[:,0],V[:,1]); return np.r_[r,r], np.r_[V[:,2],-V[:,2]]
body=pts("out/work/cage_body.stl"); legs=pts("out/work/cage_legs.stl")
HALF=body[1].max()
ZC = HALF+END_CLEAR      # 空洞の上下端
ZO = ZC+WALL             # 殻の上下端

zs=np.arange(-ZO-2, ZO+2, 0.1)
def env(p,off):
    ri,zi=p; d2=off*off-(zs[:,None]-zi[None,:])**2
    return np.where(d2>0, ri[None,:]+np.sqrt(np.maximum(d2,0)),0).max(1)
def cone45(R):                       # 45度コーンで膨張 -> |dR/dz|<=1 を保証、かつ R 以上
    return np.array([ (R - MAXSLOPE*np.abs(zs-z)).max() for z in zs ])
# 空洞＝本体の包絡面＋たわみ代 と 保持脚の包絡面＋LEG_CLR（全周の環状溝。爪の下の段が cage を吊る）
Rbody= cone45(env(body,CLEAR))
Rcav = np.maximum(Rbody, cone45(env(legs,LEG_CLR)))
# 外形は本体の空洞プロファイルの法線オフセット（肉厚を 45 度領域でも保つ）。爪の溝は中実の肉に彫るだけで外形には効かせない
d2 = WALL*WALL-(zs[:,None]-zs[None,:])**2
Rout = np.where(d2>0, Rbody[None,:]+np.sqrt(np.maximum(d2,0)), 0).max(1)

NECK_R = Rout[np.argmin(np.abs(zs))]
MAJ_R  = NECK_R+TDEPTH
BORE_R = NECK_R+TCLR
SKIRT_R= MAJ_R+TCLR+SKWALL

# ---------- 外形（上下共通） ----------
# 外形は z=0 について上下対称。|z|=SEAM_GAP からの距離 d の関数 f(d) を top は z=SEAM_GAP+d、bottom は z=-SEAM_GAP-d に置く。
# bottom の肩は z=-SEAM_GAP にあるので、top はスカート下端 z=0..SEAM_GAP が半径 SKIRT_R の垂直な帯になる（f は合わせ目で縦接線なので段は出ない）。
# f: 合わせ目で縦接線・半径 SKIRT_R の超楕円 R=SKIRT_R*(1-t^n)^(1/n) → 45 度の接線円錐 → 端面 φEND_D
# top の肩はめねじ溝＋WALL を包む必要があり、普通の楕円(n=2)では溝の上端で肉が足りない。
# 肩を張らせる指数 n を、上下両方の必要形状を包む最小値に二分法で決める（楕円の高さ b は端面径から決まる）。
g=TCLR; HG=CRESTZ/2+g+TDEPTH+2*g
GRV=np.array([(NECK_R-g,-HG),(MAJ_R+g,-CRESTZ/2-g),(MAJ_R+g,CRESTZ/2+g),(NECK_R-g,HG)])
P=[(BORE_R,z) for z in np.arange(-1,NECK_H+1e-9,0.05)]                      # めねじの下穴の壁
for c in np.arange(THR_Z0-PITCH, THR_Z0-PITCH+GRV_TURNS*PITCH+1e-9, 0.05):    # めねじ溝が通る範囲
    P+=[p0+(p1-p0)*t+[0,c] for p0,p1 in zip(GRV[:-1],GRV[1:]) for t in np.linspace(0,1,20)]
P=np.array(P); d2=WALL*WALL-(zs[:,None]-P[None,:,1])**2
Rthr=np.where(d2>0, P[None,:,0]+np.sqrt(np.maximum(d2,0)), 0).max(1)           # 溝・下穴＋WALL の包絡
ZB=-ZO                                                                         # bottom の底面
DL=ZO-SEAM_GAP; dd=np.linspace(0,DL,int(round(DL/0.1))+1)
Rneed=np.maximum.reduce([np.interp(SEAM_GAP+dd,zs,np.maximum(Rout,Rthr)), np.interp(-SEAM_GAP-dd,zs,Rout)])
REND=END_D/2
def egg(b,n):
    t=np.clip(dd/b,0,1); R=SKIRT_R*(1-t**n)**(1/n)
    w=np.where((-np.diff(R)/np.diff(dd)>=MAXSLOPE)&(R[:-1]>0))[0]               # 45 度の接点（その手前の区間はすべて 45 度未満）
    return R if len(w)==0 else np.where(dd>dd[w[0]], R[w[0]]-(dd-dd[w[0]])*MAXSLOPE, R)
def fit_b(n):                                   # 端面が φEND_D になる b
    lo,hi=5.0,300.0
    for _ in range(60):
        b=(lo+hi)/2; lo,hi=(lo,b) if egg(b,n)[-1]>REND else (b,hi)
    return b
lo,hi=2.0,8.0
for _ in range(40):
    n=(lo+hi)/2; lo,hi=(lo,n) if (egg(fit_b(n),n)-Rneed).min()>=0 else (n,hi)
EGG_N=hi; EGG_B=fit_b(EGG_N); Rprof=egg(EGG_B,EGG_N)
Regg=np.interp(np.abs(zs)-SEAM_GAP,dd,Rprof)
assert (Rthr<=SKIRT_R)[(zs>=0)&(zs<=SEAM_GAP)].all(), "top の帯がめねじ溝を包んでいない"
k=(zs>=ZB)&(zs<=ZO)
print(f"egg n{EGG_N:.2f} b{EGG_B:.2f} | margin {(Rprof-Rneed).min():.2f} at d{dd[np.argmin(Rprof-Rneed)]:.1f}"
      f" | max |dR/dz| {np.abs(np.diff(Rprof)/np.diff(dd)).max():.2f}")
Rcav0=Rbody.max()                # 首の内径（本体の空洞の最大半径。爪の溝は含めない）
gz=k&(Rcav>Rbody+1e-6)&((zs<-NECK_H)|(zs>NECK_H))     # 溝の部分の残り肉厚（首の高さはねじ側で別管理）
print(f"   groove z {zs[gz].min():.1f}..{zs[gz].max():.1f} (mirrored), r<= {Rcav[gz].max():.2f}, wall left {(Regg-Rcav)[gz].min():.2f} mm")
assert (Regg-Rcav)[gz].min()>=WALL, "爪の溝で殻が薄くなりすぎる"
print(f"=> OD {Regg[k].max()*2:.1f} mm, H {ZO-ZB:.1f} mm, base dia {np.interp(ZB,zs,Regg)*2:.1f} / top dia {np.interp(ZO,zs,Regg)*2:.1f} mm")
def rev(R,sec=256):
    k=(R>1e-6)&(zs>=ZB-0.001)&(zs<=ZO+0.001); z,r=zs[k],R[k]
    return trimesh.creation.revolve(np.column_stack([np.r_[0.,r,0.],np.r_[z[0],z,z[-1]]]),sections=sec)
def cyl(r,z0,z1,s=256):
    c=trimesh.creation.cylinder(radius=r,height=z1-z0,sections=s); c.apply_translation([0,0,(z0+z1)/2]); return c
HS=lambda a,b: cyl(300,a,b,8)
clean=lambda m: B.intersection([m,trimesh.creation.box(extents=[500,500,500])],engine=E)

cav=B.intersection([rev(Rcav),HS(-ZC,ZC)],engine=E)
out=B.intersection([rev(Regg),HS(ZB,ZO)],engine=E)

# ---------- bottom ----------
bowl=B.union([B.intersection([out,HS(-300,-SEAM_GAP)],engine=E), cyl(NECK_R,-SEAM_GAP-0.1,NECK_H)],engine=E)
bowl=B.difference([bowl,cav,cyl(Rcav0,-1,NECK_H+1)],engine=E)
HT=TDEPTH+CRESTZ/2
thr=helix_solid([(NECK_R-0.6,-HT),(NECK_R,-HT),(MAJ_R,-CRESTZ/2),
                 (MAJ_R,CRESTZ/2),(NECK_R,HT),(NECK_R-0.6,HT)],
                PITCH,THR_TURNS,THR_Z0,runout_lo=0.2,runout_hi=0.2)
bowl=B.union([bowl,B.intersection([B.difference([thr,cyl(Rcav0,-50,50)],engine=E),cyl(99,0,NECK_H)],engine=E)],engine=E)
bowl=clean(B.intersection([bowl,HS(ZB,300)],engine=E)); bowl.export("out/shell_bottom.stl")

# ---------- top (printed upside down) ----------
cap=B.intersection([out,HS(0,ZO)],engine=E)
cap=B.difference([cap,cav,cyl(BORE_R,-1,NECK_H)],engine=E)
grv=helix_solid([(NECK_R-1.2,-HG),(NECK_R-g,-HG),(MAJ_R+g,-CRESTZ/2-g),
                 (MAJ_R+g,CRESTZ/2+g),(NECK_R-g,HG),(NECK_R-1.2,HG)],
                PITCH, GRV_TURNS, THR_Z0-PITCH)
print(f"   thread P{PITCH} crest_h{2*HT:.1f} groove_h{2*HG:.1f} land{PITCH-2*HG:.1f}mm engage{THR_TURNS*360:.0f}deg")
cap=clean(B.difference([cap,grv],engine=E)); cap.export("out/shell_top.stl")

for n,m in [("bottom",bowl),("top",cap)]:
    print(f"{n}: wt={m.is_watertight} bodies={m.body_count} vol={m.volume/1000:.2f}cm3 ext={np.round(m.extents,2)}")
