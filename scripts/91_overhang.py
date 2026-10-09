import trimesh, numpy as np
L=lambda f:(lambda m:(m.merge_vertices(),m.fix_normals(),m)[2])(trimesh.load(f))
for n,f,flip in [("bottom",'out/shell_bottom.stl',False),("top(flipped)",'out/shell_top.stl',True)]:
    x=L(f)
    if flip: x.apply_transform(trimesh.transformations.rotation_matrix(np.pi,[1,0,0])); x.fix_normals()
    x.apply_translation([0,0,-x.bounds[0][2]])
    zc=x.triangles_center[:,2]; nz=x.face_normals[:,2]; a=x.area_faces
    dn=(nz<-1e-6)&(zc>0.15)          # 造形台接地面を除外
    ang=np.degrees(np.arcsin(np.clip(-nz[dn],0,1)))
    bad=ang>45.5
    print(f"{n}: 接地面を除く下向き面のうち45度超 = {a[dn][bad].sum():.1f} mm2")
    if bad.any():
        c=x.triangles_center[dn][bad]; r=np.hypot(c[:,0],c[:,1])
        print(f"    位置 z {c[:,2].min():.2f}..{c[:,2].max():.2f}  r {r.min():.2f}..{r.max():.2f}  最大角 {ang[bad].max():.1f}deg")
    print(f"    接地面積 {a[(nz<-0.99)&(zc<0.15)].sum():.0f} mm2")
