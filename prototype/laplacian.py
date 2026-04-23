from basictools import *
import scipy.sparse as sp


def get_angle_cot(verts, p, a, b):
    p, a, b = verts[p], verts[a], verts[b]
    pa = a - p
    pb = b - p

    # 计算叉积的模 (对应 sin) 和 点积 (对应 cos)
    pa_pb_cos = np.dot(pa, pb)
    pa_pb_sin = np.linalg.norm(np.cross(pa, pb))

    # 避免零向量导致的除以零错误
    if pa_pb_sin < 1e-12 and abs(pa_pb_cos) < 1e-12:
        return 0.0
    if pa_pb_sin < 1e-12:
        return float('inf') if pa_pb_cos > 0 else float('-inf')

    return pa_pb_cos / pa_pb_sin


def get_laplacian(verts, adj_mat, i):
    res = np.zeros(3, dtype=np.float32)
    link = get_link(adj_mat, i)
    for index_j in range(len(link)):
        index_prev = (index_j-1) % len(link)
        index_next = (index_j+1) % len(link)
        prev = link[index_prev]
        j = link[index_j]
        next = link[index_next]

        w = get_angle_cot(verts, prev, i, j) + \
            get_angle_cot(verts, next, i, j)
        res += w * (verts[j] - verts[i])
    return res / 2


def _get_link_fast(edge_to_opposite, adj_mat, p):
    res = []
    unordered_link = deepcopy(adj_mat[p])

    q = unordered_link.pop()
    res.append(q)

    nxt = edge_to_opposite[(min(p, q), max(p, q))][0]
    res.append(nxt)
    while unordered_link:
        unordered_link.remove(nxt)
        if not unordered_link:
            break
        for opposite in edge_to_opposite[(min(p, nxt), max(p, nxt))]:
            if opposite in unordered_link:
                nxt = opposite
        res.append(nxt)
    return res


def get_laplacians(verts, faces, adj_mat):
    n_verts = len(verts)
    laplacians = np.zeros((n_verts, 3), dtype=np.float32)

    # 预处理
    edge_to_opposite = get_edge_to_opposite(faces)

    for i in range(n_verts):
        res = np.zeros(3, dtype=np.float32)
        link = _get_link_fast(edge_to_opposite, adj_mat, i)
        n = len(link)

        for index_j in range(n):
            index_prev = (index_j - 1) % n
            index_next = (index_j + 1) % n
            prev = link[index_prev]
            j = link[index_j]
            next = link[index_next]

            w = get_angle_cot(verts, prev, i, j) + \
                get_angle_cot(verts, next, i, j)
            res += w * (verts[j] - verts[i])
        laplacians[i] = res / 2

    return laplacians


def get_laplacian_matrix(verts, adj_mat):
    vals = []
    rows = []
    cols = []

    V = len(verts)
    for i in range(V):
        link = get_link(adj_mat, i)
        w_sum = 0.0

        for index_j in range(len(link)):
            index_prev = (index_j - 1) % len(link)
            index_next = (index_j + 1) % len(link)
            prev = link[index_prev]
            j = link[index_j]
            next = link[index_next]

            w_ij = get_angle_cot(verts, prev, i, j) + \
                   get_angle_cot(verts, next, i, j)
            w_ij *= 0.5

            # 非对角项
            rows.append(i)
            cols.append(j)
            vals.append(w_ij)

            w_sum += w_ij

        # 对角项
        rows.append(i)
        cols.append(i)
        vals.append(-w_sum)

    L = sp.coo_matrix((vals, (rows, cols)), shape=(V, V))

    # 确认对称 是一种优化
    L = 0.5 * (L + L.T)

    return L.tocsr()


if __name__ == '__main__':
    verts, faces = resolve_input('input/kitten.obj')
    adj_mat = get_adj_mat(faces)
    L = get_laplacian_matrix(verts, adj_mat)
    print(sp.linalg.norm(L - L.T))
