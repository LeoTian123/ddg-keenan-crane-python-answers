from laplacian import *


verts, faces = resolve_input('input/kitten.obj')
adj_mat = get_adj_mat(faces)


laplacians = get_laplacians(verts, faces, adj_mat)
scale = np.empty(len(verts), dtype=np.float32)
def get_vertex_normals(verts, faces):
    v_normals = np.zeros_like(verts)
    for f in faces:
        v0, v1, v2 = verts[f]
        # 叉积得到面法线
        face_normal = np.cross(v1 - v0, v2 - v0)
        # 累加到三个顶点上（可以按面积加权，这里简化处理）
        v_normals[f] += face_normal

    # 归一化
    norms = np.linalg.norm(v_normals, axis=1, keepdims=True)
    return v_normals / (norms + 1e-9)

vertex_normals = get_vertex_normals(verts, faces)

# 2. 判断方向
for p in range(len(verts)):
    scale[p] = np.linalg.norm(laplacians[p])
    # 如果拉普拉斯向量与法线方向相反（点积为负），则设为负值
    if np.dot(laplacians[p], vertex_normals[p]) < 0:
        scale[p] *= -1

# 准备颜色
scale_abd_max = np.abs(scale).max()
normed_scale = scale / scale_abd_max
def blue_gray_red(t):
    if t < 0:
        # 蓝 -> 灰
        a = -t
        # a **= 0.5
        # (a)(0,0,1) + (1-a)(0.5,0.5,0.5)
        return (0.5-0.5*a, 0.5-0.5*a, 0.5+0.5*a)
    else:
        # 灰 -> 红
        # (a)(1,0,0) + (1-a)(0.5,0.5,0.5)
        a = t
        # a **= 0.5
        return (0.5+0.5*a, 0.5-0.5*a, 0.5-0.5*a)

colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)






L = get_laplacian_matrix(verts, adj_mat)

from scipy.sparse.linalg import eigsh

# 求最小的几个特征值
evals, evecs = eigsh(L, k=4, sigma=0, which='LM')

u = evecs[:, 1]
v = evecs[:, 2]
# z = u + 1j * v




# palette = np.array([
#     [1, 0, 0],     # 红
#     [0, 1, 0],     # 绿
#     [0, 0, 1],     # 蓝
#     [1, 1, 0],     # 黄
#     [1, 0, 1],     # 紫
#     [0, 1, 1],     # 青
#     [1, 0.5, 0],   # 橙
#     [0.5, 0.5, 0.5] # 灰
# ])
#
# # 计算象限索引
# x_sign = (verts[:, 0] >= 0).astype(int)
# y_sign = (verts[:, 1] >= 0).astype(int)
# z_sign = (verts[:, 2] >= 0).astype(int)
#
# quad_idx = x_sign * 4 + y_sign * 2 + z_sign
#
# # 得到每个点对应的颜色
# colors = palette[quad_idx]


import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))
plt.scatter(u, v, c=colors, s=5)
plt.axis("equal")
plt.title("Conformal Parameterization")
plt.show()