import trimesh, numpy as np, os

# 卵のモデル（Hügelschäffer の卵形）。軸は z、太い側が +z、中心（長さの中点）が原点。
# r(x) = B/2 * sqrt((L²-4x²)/(L²+8wx+4w²))   w: 最大径の位置のずれ（細い側の尖り具合）
# L・B は実測。w は未計測なので一般的な値を仮に置いている
EGGS = [  # 名前, 長さ L, 太さ B, w
    ("large", 58.5, 47.7, 2.0),
    ("small", 54.5, 41.3, 2.0),
]

def profile(L,B,w,n=400):
    x=np.linspace(-L/2,L/2,n); r=B/2*np.sqrt(np.clip((L*L-4*x*x)/(L*L+8*w*x+4*w*w),0,None))
    r[0]=r[-1]=0; return x,r

os.makedirs("out",exist_ok=True)
for name,L,B,w in EGGS:
    x,r=profile(L,B,w)
    m=trimesh.creation.revolve(np.column_stack([r,-x]),sections=256)   # 太い側を +z に
    m.merge_vertices(); m.fix_normals()
    m.export(f"out/egg_{name}.stl")
    print(f"{name}: L{L} B{B} w{w} -> wt={m.is_watertight} vol {m.volume/1000:.1f} cm3 (≒{m.volume/1000*1.075:.0f} g)")
