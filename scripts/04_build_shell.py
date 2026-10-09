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
print(f"cage R{ri.max():.2f} H{HALF*2:.2f} | cavity R{Rcav.max():.2f} | neck R{NECK_R:.2f} major R{MAJ_R:.2f} skirt R{SKIRT_R:.2f}")
print(f"=> OD {SKIRT_R*2:.1f} mm, H {ZO*2:.1f} mm, flat base dia {Rout[np.argmin(np.abs(zs+ZO))]*2:.1f} mm")
print(f"   max |dR/dz| cavity {np.abs(np.diff(Rcav)/0.1).max():.2f}  outer {np.abs(np.diff(Rout[(zs>-ZO)&(zs<ZO)])/0.1).max():.2f}")

def rev(R,sec=256):
    k=(R>1e-6)&(zs>=-ZO-0.001)&(zs<=ZO+0.001); z,r=zs[k],R[k]
    return trimesh.creation.revolve(np.column_stack([np.r_[0.,r,0.],np.r_[z[0],z,z[-1]]]),sections=sec)
def cyl(r,z0,z1,s=256):
    c=trimesh.creation.cylinder(radius=r,height=z1-z0,sections=s); c.apply_translation([0,0,(z0+z1)/2]); return c
HS=lambda a,b: cyl(300,a,b,8)
clean=lambda m: B.intersection([m,trimesh.creation.box(extents=[500,500,500])],engine=E)

cav=B.intersection([rev(Rcav),HS(-ZC,ZC)],engine=E)
out=B.intersection([rev(Rout),HS(-ZO,ZO)],engine=E)

# ---------- bottom ----------
bowl=B.union([B.intersection([out,HS(-300,0)],engine=E), cyl(NECK_R,0,NECK_H)],engine=E)
bowl=B.difference([bowl,cav,cyl(Rcav.max(),-1,NECK_H+1)],engine=E)
HT=TDEPTH+CRESTZ/2
thr=helix_solid([(NECK_R-0.6,-HT),(NECK_R,-HT),(MAJ_R,-CRESTZ/2),
                 (MAJ_R,CRESTZ/2),(NECK_R,HT),(NECK_R-0.6,HT)],
                PITCH,THR_TURNS,THR_Z0,runout_lo=0.2,runout_hi=0.2)
bowl=B.union([bowl,B.intersection([B.difference([thr,cyl(Rcav.max(),-50,50)],engine=E),cyl(99,0,NECK_H)],engine=E)],engine=E)
bowl=clean(B.intersection([bowl,HS(-ZO,300)],engine=E)); bowl.export("out/shell_bottom.stl")

# ---------- top (printed upside down) ----------
Rcap=np.maximum(Rout, SKIRT_R-np.maximum(0.0, zs-NECK_H)*MAXSLOPE)
cap=B.intersection([rev(Rcap),HS(0,ZO)],engine=E)
cap=B.difference([cap,cav,cyl(BORE_R,-1,NECK_H)],engine=E)
g=TCLR; HG=CRESTZ/2+g+TDEPTH+2*g
grv=helix_solid([(NECK_R-1.2,-HG),(NECK_R-g,-HG),(MAJ_R+g,-CRESTZ/2-g),
                 (MAJ_R+g,CRESTZ/2+g),(NECK_R-g,HG),(NECK_R-1.2,HG)],
                PITCH, GRV_TURNS, THR_Z0-PITCH)
print(f"   thread P{PITCH} crest_h{2*HT:.1f} groove_h{2*HG:.1f} land{PITCH-2*HG:.1f}mm engage{THR_TURNS*360:.0f}deg")
cap=clean(B.difference([cap,grv],engine=E)); cap.export("out/shell_top.stl")

for n,m in [("bottom",bowl),("top",cap)]:
    print(f"{n}: wt={m.is_watertight} bodies={m.body_count} vol={m.volume/1000:.2f}cm3 ext={np.round(m.extents,2)}")
