import numpy as np, trimesh, sys
from PIL import Image

def render(mesh, path, size=700, elev=25, azim=-50):
    v = mesh.vertices.copy(); f = mesh.faces
    v -= v.mean(0)
    a=np.radians(azim); e=np.radians(elev)
    Rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rx=np.array([[1,0,0],[0,np.cos(e),-np.sin(e)],[0,np.sin(e),np.cos(e)]])
    p=v@Rz.T@Rx.T
    scale=size*0.42/np.abs(p[:,[0,2]]).max()
    sx=p[:,0]*scale+size/2; sy=-p[:,2]*scale+size/2; depth=p[:,1]
    tri=np.stack([sx[f],sy[f]],-1); td=depth[f].mean(1)
    n=mesh.face_normals@Rz.T@Rx.T
    shade=np.clip(n@np.array([0.4,-0.7,0.6]),0.1,1)*0.85+0.15
    img=np.ones((size,size))*0.12; zb=np.full((size,size),1e9)
    order=np.argsort(-td)
    for i in order:
        t=tri[i]
        x0,x1=int(max(t[:,0].min(),0)),int(min(t[:,0].max()+1,size))
        y0,y1=int(max(t[:,1].min(),0)),int(min(t[:,1].max()+1,size))
        if x1<=x0 or y1<=y0: continue
        X,Y=np.meshgrid(np.arange(x0,x1)+.5,np.arange(y0,y1)+.5)
        d=(t[1,1]-t[2,1])*(t[0,0]-t[2,0])+(t[2,0]-t[1,0])*(t[0,1]-t[2,1])
        if abs(d)<1e-9: continue
        l1=((t[1,1]-t[2,1])*(X-t[2,0])+(t[2,0]-t[1,0])*(Y-t[2,1]))/d
        l2=((t[2,1]-t[0,1])*(X-t[2,0])+(t[0,0]-t[2,0])*(Y-t[2,1]))/d
        msk=(l1>=0)&(l2>=0)&(l1+l2<=1)
        if not msk.any(): continue
        sub=img[y0:y1,x0:x1]; zs=zb[y0:y1,x0:x1]
        upd=msk&(td[i]<zs)
        sub[upd]=shade[i]; zs[upd]=td[i]
    Image.fromarray((np.clip(img,0,1)*255).astype(np.uint8)).save(path)

if __name__=="__main__":
    for src,dst in zip(sys.argv[1::2], sys.argv[2::2]):
        render(trimesh.load(src), dst)
        print("ok", dst)
