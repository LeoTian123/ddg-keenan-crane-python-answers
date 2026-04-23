from laplacian import *


def get_hodge_star_0(verts, adj_mat):
    vals = []
    rows = []
    cols = []

    V = len(verts)
    for i in range(V):
        rows.append(i)
        cols.append(i)

        total_area = 0
        link = get_link(adj_mat, i)
        link.append(link[0])
        for index_j in range(len(link) - 1):
            index_k = index_j + 1
            j = link[index_j]
            k = link[index_k]
            total_area += get_area(verts, i, j, k)
        vals.append(total_area / 3)

    return sp.csr_matrix((vals, (rows, cols)), shape=(V, V))


def get_hodge_star_1_old(verts, edges, edge_to_opposite):
    vals = []
    rows = []
    cols = []

    E = len(edges)
    for i, edge in enumerate(edges):
        rows.append(i)
        cols.append(i)

        opposite = edge_to_opposite[edge]

        if len(opposite) != 2:
            vals.append(1)
            continue

        length_self = np.linalg.norm(verts[edge[0]] - verts[edge[1]])
        length_dual = np.linalg.norm(verts[opposite[0]] - verts[opposite[1]]) / 3
        vals.append(length_dual / length_self)

    return sp.csr_matrix((vals, (rows, cols)), shape=(E, E))


def get_hodge_star_1(verts, edges, edge_to_opposite):
    vals = []
    rows = []
    cols = []

    E = len(edges)
    for i, edge in enumerate(edges):
        rows.append(i)
        cols.append(i)

        opposite = edge_to_opposite[edge]

        if len(opposite) != 2:
            vals.append(1)
            continue

        cot_alpha = get_angle_cot(verts, opposite[0], *edge)
        cot_beta = get_angle_cot(verts, opposite[1], *edge)

        vals.append(0.5 * (cot_alpha + cot_beta))

    return sp.csr_matrix((vals, (rows, cols)), shape=(E, E))


def get_hodge_star_2(verts, faces):
    vals = []
    rows = []
    cols = []

    F = len(faces)
    for i, face in enumerate(faces):
        rows.append(i)
        cols.append(i)
        vals.append(1 / get_area(verts, *face))

    return sp.csr_matrix((vals, (rows, cols)), shape=(F, F))


def get_d_0(E, V, edges):
    vals = []
    rows = []
    cols = []

    for i, edge in enumerate(edges):
        rows.append(i)
        cols.append(edge[0])
        vals.append(1)

        rows.append(i)
        cols.append(edge[1])
        vals.append(-1)

    return sp.csr_matrix((vals, (rows, cols)), shape=(E, V))


def get_d_1(F, E, faces, edges):
    vals = []
    rows = []
    cols = []

    for i, face in enumerate(faces):
        v0, v1, v2 = face

        for vi, vj in [(v0, v1), (v1, v2), (v2, v0)]:
            rows.append(i)
            if vi < vj:
                cols.append(edges.index((vi, vj)))
                vals.append(1)
            else:
                cols.append(edges.index((vj, vi)))
                vals.append(-1)

    return sp.csr_matrix((vals, (rows, cols)), shape=(F, E))