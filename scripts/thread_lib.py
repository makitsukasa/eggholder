import numpy as np, trimesh

def helix_solid(section, pitch, turns, z0, steps_per_turn=96, runout_lo=0.0, runout_hi=0.0):
    """section: (n,2) (r,z) cross-section, CCW. Swept helically about +Z.
    runout_lo/hi: taper the radial depth to nothing over this many turns at each end."""
    sec=np.asarray(section,dtype=float); r0=sec[:,0].min()
    n=int(steps_per_turn*turns)+1
    th=np.linspace(0, 2*np.pi*turns, n)
    rings=[]
    for t in th:
        f=1.0
        if runout_lo>0: f=min(f, t/(2*np.pi*runout_lo))
        if runout_hi>0: f=min(f, (th[-1]-t)/(2*np.pi*runout_hi))
        f=min(max(f,0.08),1.0)
        r=r0+(sec[:,0]-r0)*f
        z=sec[:,1]*(0.4+0.6*f) + t/(2*np.pi)*pitch + z0
        rings.append(np.column_stack([r*np.cos(t), r*np.sin(t), z]))
    V=np.vstack(rings); k=len(sec); F=[]
    for j in range(n-1):
        a,b=j*k,(j+1)*k
        for i in range(k):
            i2=(i+1)%k
            F += [[a+i,a+i2,b+i2],[a+i,b+i2,b+i]]
    for i in range(1,k-1):
        F += [[0,i+1,i],[(n-1)*k,(n-1)*k+i,(n-1)*k+i+1]]
    m=trimesh.Trimesh(vertices=V, faces=np.array(F), process=True)
    m.fix_normals(); return m
