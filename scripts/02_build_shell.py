import trimesh, numpy as np, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
T=os.environ.get("PREVIEW_DIR")
from render import render
from thread_lib import helix_solid
E="manifold"; B=trimesh.boolean

CLEAR  = 1.0     # cage たわみ代
WALL   = 1.2     # 殻の肉厚
SKWALL = 1.6     # 蓋スカートの肉厚
MAXSLOPE = 1.0   # dR/dz 上限 = 45 度（オーバーハング禁止）
PITCH, TDEPTH, TCLR, CRESTZ = 5.0, 1.0, 0.30, 0.6
NECK_H, THR_Z0, THR_TURNS, GRV_TURNS = 9.5, 1.3, 1.1, 2.1
SEAM_GAP = 0.5   # 締め切った時の蓋スカート下端と bottom の肩の隙間（シールは首の頂面で取る）
END_D = 22.0     # 天面・底面の径

u=trimesh.load("out/cage_unit.stl")
V,F=trimesh.remesh.subdivide_to_size(u.vertices,u.faces,max_edge=0.4)
ri=np.hypot(V[:,0],V[:,1]); zi=V[:,2]
zi=np.r_[zi,-zi]; ri=np.r_[ri,ri]; HALF=zi.max()
ZC = HALF+CLEAR          # 空洞の上下端
ZO = ZC+WALL             # 殻の上下端

zs=np.arange(-ZO-2, ZO+2, 0.1)
def env(off):
    d2=off*off-(zs[:,None]-zi[None,:])**2
    return np.where(d2>0, ri[None,:]+np.sqrt(np.maximum(d2,0)),0).max(1)
def cone45(R):                       # 45度コーンで膨張 -> |dR/dz|<=1 を保証、かつ R 以上
    return np.array([ (R - MAXSLOPE*np.abs(zs-z)).max() for z in zs ])
Rcav = cone45(env(CLEAR))
# 外形は空洞プロファイルの法線オフセット（肉厚を 45 度領域でも保つ）
d2 = WALL*WALL-(zs[:,None]-zs[None,:])**2
Rout = np.where(d2>0, Rcav[None,:]+np.sqrt(np.maximum(d2,0)), 0).max(1)

NECK_R = Rout[np.argmin(np.abs(zs))]
MAJ_R  = NECK_R+TDEPTH
BORE_R = NECK_R+TCLR
SKIRT_R= MAJ_R+TCLR+SKWALL
Rcap=np.maximum(Rout, SKIRT_R-np.maximum(0.0, zs-NECK_H)*MAXSLOPE)

# ---------- 外形 ----------
# bottom: 合わせ目で縦接線・半径 SKIRT_R の 1/4 楕円 → 45 度の接線円錐 → 底面 φEND_D（卵の下半分）
# top   : 半径 SKIRT_R の円筒（ねじ部）→ 角 → 45 度円錐 → 角 → 天面 φEND_D
k=(zs>=-ZO)&(zs<=ZO); REND=END_D/2
def bottom_profile(b):
    t=np.clip(-(zs+SEAM_GAP)/b,0,1); R=SKIRT_R*np.sqrt(1-t*t)
    i=np.where((np.gradient(R,zs)<=MAXSLOPE)&(zs<=-SEAM_GAP)&(R>0))[0][0]   # 45 度の接点
    return np.where(zs<zs[i], R[i]+(zs-zs[i])*MAXSLOPE, R)
lo,hi=10.0,200.0                                  # 底面が φEND_D になる楕円の高さ b を二分法で
for _ in range(60):
    eb=(lo+hi)/2; lo,hi=(lo,eb) if np.interp(-ZO,zs,bottom_profile(eb))>REND else (eb,hi)
Rbot=bottom_profile(eb)
Rtop=np.minimum(SKIRT_R, REND+(ZO-zs)*MAXSLOPE)
# 円筒と円錐の角を丸める（両面に接する円弧。円筒径・円錐の位置は変えない）。
# 円筒側の接点がねじ首の上端 NECK_H に来る半径が上限（それより下はめねじ溝の外側の肉なので削れない）。
# 角から両接点までの距離は R*tan(22.5度) = R*(sqrt2-1)。
TOP_FILLET = (ZO-(SKIRT_R-REND)/MAXSLOPE - NECK_H)/(np.sqrt(2)-1)
fc_r=SKIRT_R-TOP_FILLET; fc_z=REND+ZO-fc_r-TOP_FILLET*np.sqrt(2)
fz=(zs>=fc_z)&(zs<=fc_z+TOP_FILLET/np.sqrt(2))
Rtop=np.where(fz, fc_r+np.sqrt(np.maximum(TOP_FILLET**2-(zs-fc_z)**2,0)), Rtop)
Regg=np.where(zs<0, Rbot, Rtop)
Rcav0=Rcav.max()                 # 首の内径（空洞の最大半径）
m=np.where(zs<0,Rout,Rcap)
assert (Regg-m)[k&((zs<-SEAM_GAP)|(zs>=0))].min()>=-1e-6, "外形が必要形状を包んでいない"
print(f"bottom ellipse b{eb:.2f} | top cone from z{ZO-(SKIRT_R-REND)/MAXSLOPE:.2f} fillet R{TOP_FILLET:.2f} z{fc_z:.2f}..{fc_z+TOP_FILLET/np.sqrt(2):.2f} | margin {(Regg-m)[k&((zs<-SEAM_GAP)|(zs>=0))].min():.2f}"
      f" | max |dR/dz| bottom {np.abs(np.diff(Rbot[k&(zs<-SEAM_GAP)])/0.1).max():.2f}")
print(f"=> OD {Regg[k].max()*2:.1f} mm, H {ZO*2:.1f} mm, base dia {np.interp(-ZO,zs,Regg)*2:.1f} / top dia {np.interp(ZO,zs,Regg)*2:.1f} mm")
def rev(R,sec=256):
    k=(R>1e-6)&(zs>=-ZO-0.001)&(zs<=ZO+0.001); z,r=zs[k],R[k]
    return trimesh.creation.revolve(np.column_stack([np.r_[0.,r,0.],np.r_[z[0],z,z[-1]]]),sections=sec)
def cyl(r,z0,z1,s=256):
    c=trimesh.creation.cylinder(radius=r,height=z1-z0,sections=s); c.apply_translation([0,0,(z0+z1)/2]); return c
HS=lambda a,b: cyl(300,a,b,8)
clean=lambda m: B.intersection([m,trimesh.creation.box(extents=[500,500,500])],engine=E)

cav=B.intersection([rev(Rcav),HS(-ZC,ZC)],engine=E)
out=B.intersection([rev(Regg),HS(-ZO,ZO)],engine=E)

# ---------- bottom ----------
bowl=B.union([B.intersection([out,HS(-300,-SEAM_GAP)],engine=E), cyl(NECK_R,-SEAM_GAP-0.1,NECK_H)],engine=E)
bowl=B.difference([bowl,cav,cyl(Rcav0,-1,NECK_H+1)],engine=E)
HT=TDEPTH+CRESTZ/2
thr=helix_solid([(NECK_R-0.6,-HT),(NECK_R,-HT),(MAJ_R,-CRESTZ/2),
                 (MAJ_R,CRESTZ/2),(NECK_R,HT),(NECK_R-0.6,HT)],
                PITCH,THR_TURNS,THR_Z0,runout_lo=0.2,runout_hi=0.2)
bowl=B.union([bowl,B.intersection([B.difference([thr,cyl(Rcav0,-50,50)],engine=E),cyl(99,0,NECK_H)],engine=E)],engine=E)
bowl=clean(B.intersection([bowl,HS(-ZO,300)],engine=E)); bowl.export("out/shell_bottom.stl")

# ---------- top (printed upside down) ----------
cap=B.intersection([out,HS(0,ZO)],engine=E)
cap=B.difference([cap,cav,cyl(BORE_R,-1,NECK_H)],engine=E)
g=TCLR; HG=CRESTZ/2+g+TDEPTH+2*g
grv=helix_solid([(NECK_R-1.2,-HG),(NECK_R-g,-HG),(MAJ_R+g,-CRESTZ/2-g),
                 (MAJ_R+g,CRESTZ/2+g),(NECK_R-g,HG),(NECK_R-1.2,HG)],
                PITCH, GRV_TURNS, THR_Z0-PITCH)
print(f"   thread P{PITCH} crest_h{2*HT:.1f} groove_h{2*HG:.1f} land{PITCH-2*HG:.1f}mm engage{THR_TURNS*360:.0f}deg")
cap=clean(B.difference([cap,grv],engine=E)); cap.export("out/shell_top.stl")

for n,m in [("bottom",bowl),("top",cap)]:
    print(f"{n}: wt={m.is_watertight} bodies={m.body_count} vol={m.volume/1000:.2f}cm3 ext={np.round(m.extents,2)}")
