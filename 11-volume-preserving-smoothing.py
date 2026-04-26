from closed_surface import *
from pyvista_wrapped import *

tmp = ClosedSurface('input/kitten.obj')

for _ in range(5):
    v = tmp.volume_preserving_smooth(steps=10)
    tmp = ClosedSurface.from_surface_with_new_verts(tmp, v)

    scale = tmp.get_laplacian_color_scale()
    normed_scale = scale / np.abs(scale).max()
    colors = np.array([blue_gray_red(t) for t in normed_scale], dtype=np.float32)
    vista_colored_mesh(tmp.verts, tmp.faces, colors, show_edges=False, opacity=1.0)