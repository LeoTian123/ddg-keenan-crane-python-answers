from closed_surface import *
from pyvista_wrapped import *

cs = ClosedSurface('input/kitten.obj')

print(cs.angle_defects.sum()/2/pi)
print(cs.euler_chi_by_angle_defects)

# coloring by angle_defects
scale = cs.angle_defects
normed_scale = scale / np.abs(scale).max()
colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)
vista_colored_mesh(cs.verts, cs.faces, colors, show_edges=False, opacity=1.0)


# coloring by laplacians
scale = cs.get_laplacian_color_scale()
normed_scale = scale / np.abs(scale).max()
colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)
vista_colored_mesh(cs.verts, cs.faces, colors, show_edges=False, opacity=1.0)


v = cs.backward_euler_smooth(steps=50)
# v = cs.forward_euler_smooth_origin(steps=50)
# v = cs.forward_euler_smooth_stable(steps=10)
tmp = ClosedSurface.from_surface_with_new_verts(cs, v)
scale = tmp.get_laplacian_color_scale()
normed_scale = scale / np.abs(scale).max()
colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)
vista_colored_mesh(v, cs.faces, colors, show_edges=False, opacity=1.0)