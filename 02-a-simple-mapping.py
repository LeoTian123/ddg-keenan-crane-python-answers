from closed_surface import *
from pyvista_wrapped import *
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

cs = ClosedSurface('input/double-torus.obj')

# coloring by laplacians
scale = cs.get_laplacian_color_scale()
normed_scale = scale / np.abs(scale).max()
colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)


L = cs.laplacian_matrix

# Find the smallest few eigenvalues
evals, evecs = eigsh(L, k=4, sigma=0, which='LM')

u = evecs[:, 1]
v = evecs[:, 2]
# z = u + 1j * v

# # If stained according to the quadrants:
# palette = np.array([
#     [1, 0, 0],     # red
#     [0, 1, 0],     # green
#     [0, 0, 1],     # blue
#     [1, 1, 0],     # yellow
#     [1, 0, 1],     # purple
#     [0, 1, 1],     # cyan
#     [1, 0.5, 0],   # orange
#     [0.5, 0.5, 0.5] # gray
# ])
#
# x_sign = (cs.verts[:, 0] >= 0).astype(int)
# y_sign = (cs.verts[:, 1] >= 0).astype(int)
# z_sign = (cs.verts[:, 2] >= 0).astype(int)
#
# quad_idx = x_sign * 4 + y_sign * 2 + z_sign
#
# colors = palette[quad_idx]

plt.figure(figsize=(6,6))
plt.scatter(u, v, c=colors, s=5)
plt.axis("equal")
plt.title("Conformal Parameterization")
plt.show()