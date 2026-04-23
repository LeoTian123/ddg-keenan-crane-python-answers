import pyvista as pv
import numpy as np


def blue_gray_red(t, power=1):
    if t < 0:
        # 蓝 -> 灰
        # (a)(0,0,1) + (1-a)(0.5,0.5,0.5)
        a = -t
        a **= power
        return (0.5-0.5*a, 0.5-0.5*a, 0.5+0.5*a)
    else:
        # 灰 -> 红
        # (a)(1,0,0) + (1-a)(0.5,0.5,0.5)
        a = t
        a **= power
        return (0.5+0.5*a, 0.5-0.5*a, 0.5-0.5*a)


def vista_pure_mesh(verts, faces):
    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        color='lightgray',
        opacity=0.3,
        show_edges=True
    )

    plotter.show()


def vista_verts_indices(verts, faces):
    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        color='lightgray',
        opacity=0.3,
        show_edges=True
    )

    # 添加点标签（显示 verts 的索引）
    plotter.add_point_labels(
        mesh.points,
        [str(i) for i in range(mesh.n_points)],
        font_size=12,
        point_size=5
    )

    plotter.show()


def vista_colored_mesh(verts, faces, colors, show_edges=True, opacity=1.0):
    if colors.shape != (len(verts), 3):
        raise ValueError(
            f"colors shape should be ({len(verts)}, 3), but got {colors.shape}"
        )

    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)
    mesh.point_data["colors"] = colors

    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        scalars="colors",
        rgb=True,
        show_edges=show_edges,
        opacity=opacity,
        smooth_shading=True,
        specular=0.2
    )
    plotter.show()

def plot_cycles(verts, faces, cycles):

    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)

    for i, cyc in enumerate(cycles):
        plotter = pv.Plotter(window_size=(900, 900))

        # ---- mesh ----
        plotter.add_mesh(
            mesh,
            color='lightgray',
            opacity=0.35,
            show_edges=True
        )

        # ---- 单个 cycle ----
        # pts = verts[[u for u, _ in cyc]]

        pts = [sum(verts[faces[index_f]])/3 for index_f in cyc]
        poly = pv.PolyData()
        poly.points = pts
        poly.lines = np.hstack([[len(pts)] + list(range(len(pts)))])

        plotter.add_mesh(
            poly,
            color='blue',
            line_width=4,
            render_lines_as_tubes=True
        )

        plotter.add_title(f"Cycle {i}", font_size=12)
        plotter.show()   # 阻塞式弹窗


def pyvista_edge_1_form(
    verts,
    faces,
    edges,
    form,
    factor=0.3,
    mesh_color="lightgray",
    arrow_color="blue",
    opacity=1.0,
    show_edges=False,
):
    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)

    plotter = pv.Plotter(window_size=(900, 900))

    plotter.add_mesh(
        mesh,
        color=mesh_color,
        opacity=opacity,
        show_edges=show_edges,
    )

    arrow_origins = []
    arrow_dirs = []

    for (u, v), w in zip(edges, form):
        p0 = verts[u]
        p1 = verts[v]

        direction = p1 - p0
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = direction / norm

        vec = w * direction
        origin = 0.5 * (p0 + p1) - 0.5 * vec * factor

        arrow_origins.append(origin)
        arrow_dirs.append(vec)

    if arrow_origins:
        pdata = pv.PolyData(np.array(arrow_origins))
        pdata["vectors"] = np.array(arrow_dirs)

        arrows = pdata.glyph(
            orient="vectors",
            scale="vectors",
            factor=factor,
        )
        plotter.add_mesh(arrows, color=arrow_color)

    plotter.add_title("1-form", font_size=12)
    plotter.show()


def pyvista_multiple_edge_1_forms(verts, faces, edges, omegas, factor=0.02):

    if not omegas:
        return

    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)

    plotter = pv.Plotter(window_size=(900, 900))

    plotter.add_mesh(
        mesh,
        color='lightgray',
        opacity=1,
        show_edges=False
    )

    state = {"idx": 0, "actor": None}

    def build_arrows(omega):
        arrow_origins = []
        arrow_dirs = []

        for (u, v), w in zip(edges, omega):
            p0 = verts[u]
            p1 = verts[v]

            mid = 0.5 * (p0 + p1)
            direction = p1 - p0
            direction = direction / np.linalg.norm(direction)

            vec = w * direction

            arrow_origins.append(mid)
            arrow_dirs.append(vec)

        pdata = pv.PolyData(np.array(arrow_origins))
        pdata["vectors"] = np.array(arrow_dirs)

        arrows = pdata.glyph(
            orient="vectors",
            scale="vectors",
            factor=factor
        )
        return arrows

    def update():
        # 删除旧的
        if state["actor"] is not None:
            plotter.remove_actor(state["actor"])

        omega = omegas[state["idx"]]
        arrows = build_arrows(omega)

        actor = plotter.add_mesh(arrows, color="blue")
        state["actor"] = actor

        plotter.add_title(f"1-form #{state['idx']}", font_size=12)
        plotter.render()

    # ---- 键盘事件 ----
    def next_form():
        state["idx"] = (state["idx"] + 1) % len(omegas)
        update()

    def prev_form():
        state["idx"] = (state["idx"] - 1) % len(omegas)
        update()

    plotter.add_key_event("n", next_form)
    plotter.add_key_event("p", prev_form)

    # 初始显示
    update()

    plotter.add_text(
        "press 'n' and 'p' to switch 1-forms.",
        position="lower_left",
        font_size=12,
        color="black"
    )
    plotter.show()


def pyvista_dual_edge_1_form(
    closed_surface, phi,
    factor=0.3,
    mesh_color="lightgray",
    arrow_color="blue",
    opacity=1.0,
    show_edges=False,
):
    """
    可视化定义在 dual edges 上的 1-form phi。

    ----
    phi   : (num_dual_edges,) ndarray
        定义在 dual edge 上的数值。
    factor : float
        箭头整体缩放系数。
    """
    verts = closed_surface.verts
    faces = closed_surface.faces
    dual_edge_to_index = closed_surface.dual_edge_to_index
    dual_edge_to_order = closed_surface.dual_edge_to_order

    # PyVista 的面格式
    face_array = np.hstack([[3, *f] for f in faces])
    mesh = pv.PolyData(verts, face_array)

    # 用 face barycenter 近似 dual vertices
    face_centers = verts[faces].mean(axis=1)   # (F, 3)

    arrow_origins = []
    arrow_dirs = []

    for (f0, f1), idx in dual_edge_to_index.items():
        w = phi[idx]

        if not (f0, f1) == dual_edge_to_order[(f0, f1)]:
            f0, f1 = f1, f0

        c0 = face_centers[f0]
        c1 = face_centers[f1]

        direction = c1 - c0
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        direction = direction / norm

        # 箭头方向沿 dual edge，正负号由 w 体现
        vec = w * direction

        # 让箭头大致居中落在 dual edge 上
        origin = 0.5 * (c0 + c1) - 0.5 * vec * factor

        arrow_origins.append(origin)
        arrow_dirs.append(vec)

    arrow_origins = np.asarray(arrow_origins)
    arrow_dirs = np.asarray(arrow_dirs)

    pdata = pv.PolyData(arrow_origins)
    pdata["vectors"] = arrow_dirs
    pdata["magnitude"] = np.linalg.norm(arrow_dirs, axis=1)

    arrows = pdata.glyph(
        orient="vectors",
        scale="magnitude",
        factor=factor,
    )

    plotter = pv.Plotter(window_size=(900, 900))
    plotter.add_mesh(
        mesh,
        color=mesh_color,
        opacity=opacity,
        show_edges=show_edges,
    )
    plotter.add_mesh(arrows, color=arrow_color)
    plotter.add_title("connection 1-form phi on dual edges", font_size=12)
    plotter.show()


def pyvista_vectors_on_faces(
    verts,
    faces,
    face_vectors,
    face_centers=None,
    vector_scale=0.08,
    show_surface=True,
    show_edges=True,
    surface_opacity=0.35,
    color="tomato",
    line_width=3.0,
    indices=[],
):
    """
    用 pyvista 可视化 face 上的向量场。
    """
    if face_centers is None:
        face_centers = verts[faces].mean(axis=1)

    # PyVista faces 格式: [3, i, j, k, 3, ...]
    faces_pv = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()

    mesh = pv.PolyData(verts, faces_pv)
    centers_cloud = pv.PolyData(face_centers)
    centers_cloud["vectors"] = face_vectors

    glyphs = centers_cloud.glyph(
        orient="vectors",
        scale=False,
        factor=vector_scale,
    )

    pl = pv.Plotter()

    if show_surface:
        pl.add_mesh(
            mesh,
            color="lightgray",
            opacity=surface_opacity,
            show_edges=show_edges,
        )

    # 取出这些点的坐标
    if indices:
        selected_points = mesh.points[indices]
        # 对应标签
        labels = [str(i) for i in indices]
        pl.add_point_labels(
            selected_points,
            labels,
            font_size=12,
            point_size=5
        )

    pl.add_mesh(glyphs, color=color, line_width=line_width)
    pl.add_axes()
    pl.show()


if __name__ == '__main__':
    from basictools import *
    verts, faces = resolve_input('input/sphere.obj')
    # vista_pure_mesh(verts, faces)
    vista_verts_indices(verts, faces)