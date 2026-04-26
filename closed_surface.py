import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import math
pi = math.pi

from collections import defaultdict, deque
from itertools import pairwise
from copy import deepcopy

from tqdm import tqdm


class ClosedSurface:
    def __init__(self, file_path):
        verts = []
        faces = []
        with open(file_path, 'r') as f:
            for line in f:
                if line[0] == 'v':
                    vert = list(map(float, line.split()[1:]))
                    verts.append(vert)
                elif line[0] == 'f':
                    face = list(map(int, line.split()[1:]))
                    faces.append(face)
        self.verts = np.array(verts, dtype=np.float64)
        self.faces = np.array(faces, dtype=np.int32) - 1
        self.V = len(self.verts)
        self._E = None
        self.F = len(self.faces)

        self._adj_mat = None
        self._co_adj_mat = None

        # about "edges_preprocess"
        self._edges = None
        self._dual_edges = None
        self._edge_to_opposite = None
        self._edge_to_dual_edge = None
        self._dual_edge_to_edge = None
        self._edge_to_index = None
        self._dual_edge_to_index = None
        self._dual_edge_to_order = None

        # about Gaussian curvature
        self._angle_defects = None
        self._euler_chi_by_angle_defects = None

        # about laplacian
        self._laplacians = None
        self._laplacian_matrix = None

        # about chapter 8
        self._star_0 = None
        self._star_1 = None
        self._star_2 = None
        self._d_0 = None
        self._d_1 = None

    @classmethod
    def from_surface_with_new_verts(cls, surface, verts):
        # Mainly used in processes about euler_smooth.
        obj = cls.__new__(cls)

        fields_set = {
            'faces',
            'V', '_E', 'F',

            '_adj_mat', '_co_adj_mat',

            '_edges', '_dual_edges',
            '_edge_to_opposite',
            '_edge_to_dual_edge', '_dual_edge_to_edge',
            '_edge_to_index', '_dual_edge_to_index',
            '_dual_edge_to_order',
        }

        for k, v in surface.__dict__.items():
            if k in fields_set:
                setattr(obj, k, deepcopy(v))
            else:
                setattr(obj, k, None)

        obj.verts = verts.copy()

        return obj

    def get_angle(self, p, a, b):
        p, a, b = self.verts[p], self.verts[a], self.verts[b]
        pa = a - p
        pb = b - p
        cross_norm = np.linalg.norm(np.cross(pa, pb))
        dot_val = np.dot(pa, pb)
        return math.atan2(cross_norm, dot_val)

    def get_normal(self, v0, v1, v2):
        v0, v1, v2 = self.verts[v0], self.verts[v1], self.verts[v2]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n) + 1e-12
        return n

    def get_area(self, v0, v1, v2):
        v0, v1, v2 = self.verts[v0], self.verts[v1], self.verts[v2]
        n = np.cross(v1 - v0, v2 - v0)
        return np.linalg.norm(n)

    def get_volume(self):
        res = 0
        for i, j, k in self.faces:
            res += np.dot(self.verts[i], np.cross(self.verts[j], self.verts[k]))
        return res

    def get_surface_area(self):
        res = 0
        for f in self.faces:
            res += self.get_area(*f)
        return res

    @property
    def adj_mat(self):
        if self._adj_mat is None:
            adj_mat = defaultdict(set)
            for face in self.faces:
                v0, v1, v2 = face
                adj_mat[v0].add(v1)
                adj_mat[v1].add(v0)
                adj_mat[v1].add(v2)
                adj_mat[v2].add(v1)
                adj_mat[v2].add(v0)
                adj_mat[v0].add(v2)
            self._adj_mat = adj_mat
        return self._adj_mat

    def _ensure_edges_preprocessed(self):
        if self._edges is not None:
            return

        edge_to_opposite = defaultdict(list)
        edge_to_dual_edge = defaultdict(list)
        dual_edge_to_edge = {}

        edge_to_index = {}
        dual_edge_to_index = {}

        # edge_to_opposite, 无序的 edge_to_dual_edge
        for index_f, (v0, v1, v2) in enumerate(self.faces):
            v0, v1, v2 = sorted([v0, v1, v2])
            edge_to_opposite[(v0, v1)].append(v2)
            edge_to_opposite[(v0, v2)].append(v1)
            edge_to_opposite[(v1, v2)].append(v0)

            edge_to_dual_edge[(v0, v1)].append(index_f)
            edge_to_dual_edge[(v0, v2)].append(index_f)
            edge_to_dual_edge[(v1, v2)].append(index_f)

        # 有序的 edge_to_dual_edge, dual_edge_to_edge
        for edge, dual_edges in edge_to_dual_edge.items():
            edge_to_dual_edge[edge].sort()
            dual_edge = edge_to_dual_edge[edge]
            dual_edge = tuple(dual_edge)

            edge_to_dual_edge[edge] = dual_edge
            dual_edge_to_edge[dual_edge] = edge

        # edges, dual_edges
        edges = list(edge_to_dual_edge.keys())
        dual_edges = list(dual_edge_to_edge.keys())

        # edge_to_index
        for i, edge in enumerate(edges):
            edge_to_index[edge] = i

        # dual_edge_to_index
        for i, dual_edge in enumerate(dual_edges):
            dual_edge_to_index[dual_edge] = i

        self._edges = edges
        self._dual_edges = dual_edges

        self._edge_to_opposite = dict(edge_to_opposite)
        self._edge_to_dual_edge = dict(edge_to_dual_edge)
        self._dual_edge_to_edge = dual_edge_to_edge

        self._edge_to_index = edge_to_index
        self._dual_edge_to_index = dual_edge_to_index

    def order_of_dual_edge(self, edge):
        two_neighbor_faces = self.edge_to_dual_edge[edge]
        a_neighbor_face = two_neighbor_faces[0]
        a, b, c = self.faces[a_neighbor_face]
        u, v = edge

        if (a, b) == (u, v) or (b, c) == (u, v) or (c, a) == (u, v):
            left_face = a_neighbor_face
            right_face = two_neighbor_faces[1]
        else:
            left_face = two_neighbor_faces[1]
            right_face = a_neighbor_face

        return left_face, right_face

    @property
    def dual_edge_to_order(self):
        if self._dual_edge_to_order is None:
            dual_edge_to_order = {}
            for dual_edge in self.dual_edges:
                edge = self.dual_edge_to_edge[dual_edge]
                dual_edge_to_order[dual_edge] = self.order_of_dual_edge(edge)
            self._dual_edge_to_order = dual_edge_to_order
        return self._dual_edge_to_order

    @property
    def edges(self):
        self._ensure_edges_preprocessed()
        return self._edges

    @property
    def dual_edges(self):
        self._ensure_edges_preprocessed()
        return self._dual_edges

    @property
    def edge_to_opposite(self):
        self._ensure_edges_preprocessed()
        return self._edge_to_opposite

    @property
    def edge_to_dual_edge(self):
        self._ensure_edges_preprocessed()
        return self._edge_to_dual_edge

    @property
    def dual_edge_to_edge(self):
        self._ensure_edges_preprocessed()
        return self._dual_edge_to_edge

    @property
    def edge_to_index(self):
        self._ensure_edges_preprocessed()
        return self._edge_to_index

    @property
    def dual_edge_to_index(self):
        self._ensure_edges_preprocessed()
        return self._dual_edge_to_index

    @property
    def E(self):
        if self._E is None:
            self._E = len(self.edges)
        return self._E

    @property
    def co_adj_mat(self):
        if self._co_adj_mat is None:
            co_adj_mat = defaultdict(set)
            for dual_edge in self.dual_edges:
                f0, f1 = dual_edge
                co_adj_mat[f0].add(f1)
                co_adj_mat[f1].add(f0)
            self._co_adj_mat = co_adj_mat
        return self._co_adj_mat

    def get_link(self, p):
        res = []
        unordered_link = deepcopy(self.adj_mat[p])

        q = unordered_link.pop()
        res.append(q)

        # ... --- last --- q=first --- next=second --- ...
        # second, last = list(unordered_link & adj_mat[q])

        nxt = list(unordered_link & self.adj_mat[q])[0]
        res.append(nxt)
        try:
            while unordered_link:
                unordered_link.remove(nxt)
                if not unordered_link:
                    break
                nxt = list(unordered_link & self.adj_mat[nxt])[0]
                res.append(nxt)
            return res
        except Exception:
            return res

    def _get_link_fast(self, p):
        res = []
        unordered_link = deepcopy(self.adj_mat[p])

        q = unordered_link.pop()
        res.append(q)

        nxt = self.edge_to_opposite[(min(p, q), max(p, q))][0]
        res.append(nxt)
        while unordered_link:
            unordered_link.remove(nxt)
            if not unordered_link:
                break
            for opposite in self.edge_to_opposite[(min(p, nxt), max(p, nxt))]:
                if opposite in unordered_link:
                    nxt = opposite
            res.append(nxt)
        return res

    def angle_defect(self, p):
        res = 0.0
        for neighbor1 in self.adj_mat[p]:
            for neighbor2 in self.adj_mat[p]:
                if neighbor1 in self.adj_mat[neighbor2]:
                    res += self.get_angle(p, neighbor1, neighbor2)
        res = 4 * pi - res
        return res / 2

    @property
    def angle_defects(self):
        if self._angle_defects is None:
            angle_defects = np.full(self.V, 2 * pi, dtype=np.float64)

            for i, j, k in self.faces:
                angle_defects[i] -= self.get_angle(i, j, k)
                angle_defects[j] -= self.get_angle(j, k, i)
                angle_defects[k] -= self.get_angle(k, i, j)

            self._angle_defects = angle_defects

        return self._angle_defects

    @property
    def euler_chi_by_angle_defects(self):
        if self._euler_chi_by_angle_defects is None:
            self._euler_chi_by_angle_defects = round(
                self.angle_defects.sum() / (2 * pi)
            )
        return self._euler_chi_by_angle_defects

    def get_angle_cot(self, p, a, b):
        p, a, b = self.verts[p], self.verts[a], self.verts[b]
        pa = a - p
        pb = b - p

        # culculate the norm of cross-product (sin) and dot-product (对应 cos)
        pa_pb_cos = np.dot(pa, pb)
        pa_pb_sin = np.linalg.norm(np.cross(pa, pb))

        if pa_pb_sin < 1e-12 and abs(pa_pb_cos) < 1e-12:
            return 0.0
        if pa_pb_sin < 1e-12:
            return float('inf') if pa_pb_cos > 0 else float('-inf')

        return pa_pb_cos / pa_pb_sin

    def laplacian(self, i):
        res = np.zeros(3, dtype=np.float64)
        link = self.get_link(i)
        for index_j in range(len(link)):
            index_prev = (index_j - 1) % len(link)
            index_next = (index_j + 1) % len(link)
            prev = link[index_prev]
            j = link[index_j]
            next = link[index_next]

            w = self.get_angle_cot(prev, i, j) + self.get_angle_cot(next, i, j)
            res += w * (self.verts[j] - self.verts[i])
        return res / 2

    @property
    def laplacians(self):
        if self._laplacians is None:
            laplacians = np.zeros((self.V, 3), dtype=np.float64)

            for i in range(self.V):
                res = np.zeros(3, dtype=np.float64)
                link = self._get_link_fast(i)

                for index_j in range(len(link)):
                    index_prev = (index_j - 1) % len(link)
                    index_next = (index_j + 1) % len(link)
                    prev = link[index_prev]
                    j = link[index_j]
                    next = link[index_next]

                    w = self.get_angle_cot(prev, i, j) + self.get_angle_cot(next, i, j)
                    res += w * (self.verts[j] - self.verts[i])

                laplacians[i] = res / 2

            self._laplacians = laplacians

        return self._laplacians

    @property
    def laplacian_matrix(self):
        if self._laplacian_matrix is None:
            vals = []
            rows = []
            cols = []

            for i in range(self.V):
                link = self._get_link_fast(i)
                w_sum = 0.0
                n = len(link)

                for index_j in range(n):
                    index_prev = (index_j - 1) % n
                    index_next = (index_j + 1) % n
                    prev = link[index_prev]
                    j = link[index_j]
                    next = link[index_next]

                    w_ij = self.get_angle_cot(prev, i, j) + self.get_angle_cot(next, i, j)
                    w_ij *= 0.5

                    rows.append(i)
                    cols.append(j)
                    vals.append(w_ij)

                    w_sum += w_ij

                rows.append(i)
                cols.append(i)
                vals.append(-w_sum)

            L = sp.coo_matrix((vals, (rows, cols)), shape=(self.V, self.V))
            L = 0.5 * (L + L.T)
            self._laplacian_matrix = L.tocsr()

        return self._laplacian_matrix

    def get_laplacian_color_scale(self):
        scale = np.empty(self.V, dtype=np.float64)

        # 1. rough vertex_normals by adding neighbor face_normals
        vertex_normals = np.zeros_like(self.verts)
        for f in self.faces:
            face_normal = self.get_normal(*f)
            vertex_normals[f] += face_normal
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals /= norms + 1e-12

        # 2. give the magnitude and direction one-by-one
        for p in range(self.V):
            scale[p] = np.linalg.norm(self.laplacians[p])
            if np.dot(self.laplacians[p], vertex_normals[p]) < 0:
                scale[p] *= -1

        return scale

    def forward_euler_smooth_stable(self, steps=50, lam=0.1):
        v = self.verts.copy()
        for _ in tqdm(range(steps), desc="Forward Euler Smooth Stable"):
            tmp = ClosedSurface.from_surface_with_new_verts(self, v)
            lplc = tmp.laplacians
            v = v + lam * lplc
        return v

    def forward_euler_smooth_origin(self, steps=50, lam=0.1):
        v = self.verts.copy()
        laplacians = self.laplacians.copy()
        for _ in tqdm(range(steps), desc="Forward Euler Smooth Origin"):
            v = v + lam * laplacians
        return v

    def backward_euler_smooth(self, steps=50, lam=0.1):
        v = self.verts.copy()
        L = self.laplacian_matrix
        n = L.shape[0]
        I = sp.eye(n)
        A = (I - lam * L).tocsc()
        solver = sp.linalg.factorized(A)

        for _ in tqdm(range(steps), desc="Backward Euler Smooth"):
            for d in range(3):
                v[:, d] = solver(v[:, d])
        return v

    def volume_preserving_smooth(self, steps=10, lam=0.1):
        def get_one_step_shift(cs):
            ui = cs.laplacians
            gradient_volume = np.zeros_like(cs.verts)
            for i, j, k in cs.faces:
                gradient_volume[i] += np.cross(cs.verts[j], cs.verts[k])
                gradient_volume[j] += np.cross(cs.verts[k], cs.verts[i])
                gradient_volume[k] += np.cross(cs.verts[i], cs.verts[j])
            gradient_volume /= 6

            projection_magnitude = (
                    np.sum(ui * gradient_volume) /
                    np.sum(gradient_volume * gradient_volume)
            )

            return lam * (ui - projection_magnitude * gradient_volume)

        v = self.verts.copy()
        tmp = ClosedSurface.from_surface_with_new_verts(self, v)
        volume0 = tmp.get_volume()
        for _ in tqdm(range(steps), desc="volume_maintained_smooth"):
            shift = get_one_step_shift(tmp)
            v = v + shift

            tmp = ClosedSurface.from_surface_with_new_verts(self, v)
            volume = tmp.get_volume()

            bary_center = np.mean(tmp.verts, axis=0)
            v = bary_center + (volume0 / volume) ** (1 / 3) * (v - bary_center)

            tmp = ClosedSurface.from_surface_with_new_verts(self, v)

        return v

    def volume_maintained_smooth_simple(self, steps=10, lam=0.1):
        v = self.verts.copy()
        tmp = ClosedSurface.from_surface_with_new_verts(self, v)
        volume0 = tmp.get_volume()
        for _ in tqdm(range(steps), desc="volume_maintained_smooth_simple"):
            shift = lam * tmp.laplacians
            v = v + shift

            tmp = ClosedSurface.from_surface_with_new_verts(self, v)
            volume = tmp.get_volume()

            bary_center = np.mean(tmp.verts, axis=0)
            v = bary_center + (volume0 / volume) ** (1 / 3) * (v - bary_center)

            tmp = ClosedSurface.from_surface_with_new_verts(self, v)

        return v

    @property
    def star_0(self):
        if self._star_0 is None:
            vals = []
            rows = []
            cols = []

            for i in range(self.V):
                rows.append(i)
                cols.append(i)

                total_area = 0.0
                link = self._get_link_fast(i)
                link.append(link[0])
                for index_j in range(len(link) - 1):
                    j = link[index_j]
                    k = link[index_j + 1]
                    total_area += self.get_area(i, j, k)
                vals.append(total_area / 3)

            self._star_0 = sp.csr_matrix((vals, (rows, cols)), shape=(self.V, self.V))

        return self._star_0

    @property
    def star_1(self):
        if self._star_1 is None:
            vals = []
            rows = []
            cols = []

            for i, edge in enumerate(self.edges):
                rows.append(i)
                cols.append(i)

                opposite = self.edge_to_opposite[edge]

                if len(opposite) != 2:
                    vals.append(1)
                    continue

                cot_alpha = self.get_angle_cot(opposite[0], *edge)
                cot_beta = self.get_angle_cot(opposite[1], *edge)
                vals.append(0.5 * (cot_alpha + cot_beta))

            self._star_1 = sp.csr_matrix((vals, (rows, cols)), shape=(self.E, self.E))

        return self._star_1

    @property
    def star_2(self):
        if self._star_2 is None:
            vals = []
            rows = []
            cols = []

            for i, face in enumerate(self.faces):
                rows.append(i)
                cols.append(i)
                vals.append(1 / self.get_area(*face))

            self._star_2 = sp.csr_matrix((vals, (rows, cols)), shape=(self.F, self.F))

        return self._star_2

    @property
    def d_0(self):
        if self._d_0 is None:
            vals = []
            rows = []
            cols = []

            for i, edge in enumerate(self.edges):
                rows.append(i)
                cols.append(edge[0])
                vals.append(1)

                rows.append(i)
                cols.append(edge[1])
                vals.append(-1)

            self._d_0 = sp.csr_matrix((vals, (rows, cols)), shape=(self.E, self.V))

        return self._d_0

    @property
    def d_1(self):
        if self._d_1 is None:
            vals = []
            rows = []
            cols = []

            for i, face in enumerate(self.faces):
                v0, v1, v2 = face

                for vi, vj in [(v0, v1), (v1, v2), (v2, v0)]:
                    rows.append(i)
                    if vi < vj:
                        cols.append(self.edge_to_index[(vi, vj)])
                        vals.append(1)
                    else:
                        cols.append(self.edge_to_index[(vj, vi)])
                        vals.append(-1)

            self._d_1 = sp.csr_matrix((vals, (rows, cols)), shape=(self.F, self.E))

        return self._d_1

    def get_generators(self, root=0, co_root=0):
        if not (0 <= root < self.V):
            raise IndexError(f"root index out of bounds: {root}")
        if not (0 <= co_root < self.F):
            raise IndexError(f"co_root index out of bounds: {co_root}")

        # cotree
        covertex_visited = [False] * self.F  # covertex is just face
        cotree_edges = set()

        q = deque([co_root])
        covertex_visited[co_root] = True

        while q:
            f = q.popleft()
            for neighbor in self.co_adj_mat[f]:
                if not covertex_visited[neighbor]:
                    covertex_visited[neighbor] = True
                    cotree_edges.add((min(f, neighbor), max(f, neighbor)))
                    q.append(neighbor)

        # tree的边不能和cotree的对偶边交叉，即对偶回来不能相同
        edge_selected = [self.dual_edge_to_edge[dual_edge] for dual_edge in list(cotree_edges)]
        edge_selected = set(edge_selected)

        # tree
        vertex_visited = [False] * self.V  # vertex is just vertex
        tree_edges = set()

        q = deque([root])
        vertex_visited[root] = True

        while q:
            v = q.popleft()
            for neighbor in self.adj_mat[v]:
                current_edge = (min(v, neighbor), max(v, neighbor))
                if current_edge not in edge_selected:
                    if not vertex_visited[neighbor]:
                        vertex_visited[neighbor] = True
                        tree_edges.add(current_edge)
                        q.append(neighbor)

        # 得到剩余边
        all_edge_selected = edge_selected | tree_edges
        left_edges = list(set(self.edges) - all_edge_selected)
        left_dual_edges = [self.edge_to_dual_edge[edge] for edge in left_edges]

        # 构建cotree的邻接表
        cotree_adj_mat = defaultdict(set)
        for dual_edge in cotree_edges:
            f0, f1 = dual_edge
            cotree_adj_mat[f0].add(f1)
            cotree_adj_mat[f1].add(f0)

        # 经典算法题，其实就是图中寻路
        def get_generator(left_dual_edge):
            (u, v) = left_dual_edge

            parent = dict()  # parent[x] = x 在 BFS/DFS 树中的父节点
            visited = set()
            stack = [u]
            visited.add(u)
            parent[u] = None

            # 在树里找 u 到 v 的路径
            while stack:
                x = stack.pop()
                if x == v:
                    break
                for y in cotree_adj_mat[x]:
                    if y not in visited:
                        visited.add(y)
                        parent[y] = x
                        stack.append(y)

            # 回溯 v -> u
            path = []
            cur = v
            while cur is not None:
                path.append(cur)
                cur = parent[cur]

            path.append(v)
            return path

        generators = []
        for left_dual_edge in left_dual_edges:
            generators.append(get_generator(left_dual_edge))

        return generators

    def get_harmonic_bases(self, generators):
        omegas = []
        for generator in generators:
            omega = np.zeros(self.E)
            for u, v in pairwise(generator):
                # u, v：path order
                order_by_indices = (min(u, v), max(u, v))  # to find corresponding information

                index_dual_edge = self.dual_edge_to_index[order_by_indices]
                left_face, right_face = self.dual_edge_to_order[order_by_indices]

                if left_face == u:
                    omega[index_dual_edge] = 1
                else:
                    omega[index_dual_edge] = -1

            omegas.append(omega)

        d0 = self.d_0
        star1 = self.star_1

        gammas = []
        for omega in omegas:
            A = d0.T @ star1 @ d0  # (|V| × |V|) , namely laplacian
            b = d0.T @ star1 @ omega  # (|V|) , namely co-differential
            alpha = spla.spsolve(A, b)
            gamma = omega - d0 @ alpha
            gammas.append(gamma)

        return gammas, omegas

    def integral_dual_path_dual_1_form(self, dual_path, dual_1_form):
        res = 0
        for u, v in pairwise(dual_path):
            # u, v：path order
            order_by_indices = (min(u, v), max(u, v))  # to find corresponding information

            index_dual_edge = self.dual_edge_to_index[order_by_indices]
            left_face, right_face = self.dual_edge_to_order[order_by_indices]

            if left_face == u:
                sign = 1
            else:
                sign = -1

            res += sign * dual_1_form[index_dual_edge]
        return res

    def get_P(self, generators, gammas):
        two_g = len(generators)
        P = np.zeros(shape=(two_g, two_g))
        for i in range(two_g):
            for j in range(two_g):
                P[i, j] = self.integral_dual_path_dual_1_form(generators[i], gammas[j])
        return P

    def get_holonomy_of_levi_civita_connection_on_generator(self, generator):
        """ This function is implemented by AI. """

        if len(generator) < 2:
            return 0.0

        def _normalize(x, eps=1e-12):
            n = np.linalg.norm(x)
            if n < eps:
                return x.copy()
            return x / n

        def _rotate_about_axis(v, axis, angle):
            # Rodrigues formula, axis must be unit
            c = math.cos(angle)
            s = math.sin(angle)
            return (
                    v * c
                    + np.cross(axis, v) * s
                    + axis * np.dot(axis, v) * (1 - c)
            )

        # ---- choose an initial tangent vector on the first face ----
        f0 = generator[0]
        n0 = _normalize(self.get_normal(*self.faces[f0]))

        # pick one edge direction in the first face as initial tangent vector
        a, b, c = self.faces[f0]
        w0 = self.verts[b] - self.verts[a]
        w0 = w0 - np.dot(w0, n0) * n0
        w0 = _normalize(w0)

        w = w0.copy()

        # ---- parallel transport the SAME vector across each adjacent face pair ----
        for u, v in pairwise(generator):
            order_by_indices = (min(u, v), max(u, v))
            edge = self.dual_edge_to_edge[order_by_indices]

            # shared primal edge direction
            axis = self.verts[edge[1]] - self.verts[edge[0]]
            axis = _normalize(axis)

            n_u = _normalize(self.get_normal(*self.faces[u]))
            n_v = _normalize(self.get_normal(*self.faces[v]))

            # signed angle rotating normal_u to normal_v around the shared edge axis
            phi = math.atan2(
                np.dot(axis, np.cross(n_u, n_v)),
                np.dot(n_u, n_v)
            )

            # transport the vector by rotating it around the shared edge
            w = _rotate_about_axis(w, axis, phi)

            # numerical cleanup: re-project to the tangent plane of face v
            w = w - np.dot(w, n_v) * n_v
            w = _normalize(w)

        # ---- compare final vector with initial vector in the initial tangent plane ----
        w = w - np.dot(w, n0) * n0
        w = _normalize(w)

        holonomy = math.atan2(
            np.dot(n0, np.cross(w0, w)),
            np.dot(w0, w)
        )

        return holonomy

    def get_trivial_connection(self, generators, gammas, point_indices, singularity_type_input):

        two_g = len(generators)
        euler_chi = 2 - two_g

        if sum(singularity_type_input) != euler_chi:
            raise ValueError("sum(singularity_type_input) != euler_chi")

        singularity_type = np.zeros(self.V)
        singularity_type[point_indices] = singularity_type_input

        d0 = self.d_0
        star1 = self.star_1

        u = - self.angle_defects + 2 * pi * singularity_type

        A = d0.T @ star1 @ d0  # (|V| × |V|)
        b = u  # (|V|)
        beta_tilde = spla.spsolve(A, b)
        delta_beta = star1 @ d0 @ beta_tilde


        if two_g == 0:
            phi = delta_beta

        else:
            v_tilde = np.zeros(two_g)
            for i in range(two_g):
                vi = self.get_holonomy_of_levi_civita_connection_on_generator(generators[i])
                v_tilde[i] = vi - self.integral_dual_path_dual_1_form(generators[i], delta_beta)

            P = self.get_P(generators, gammas)
            z = np.linalg.lstsq(P, v_tilde, rcond=None)[0]

            gammas_arr = np.array(gammas)
            harmonic = z @ gammas_arr

            phi = delta_beta + harmonic

        return phi, delta_beta


# ----------------------------
# The code from here on out is for generating vector fields using phi.
# Given by AI.
# ----------------------------

def normalize(x):
    return x / np.linalg.norm(x)


def rotate_axis_angle(v, axis, theta):
    axis = normalize(axis)
    c = math.cos(theta)
    s = math.sin(theta)
    return c * v + s * np.cross(axis, v) + (1.0 - c) * np.dot(axis, v) * axis


def levi_civita_transport_across_edge(v, n_u, n_v, edge_dir):
    """
    把 face u 上的切向量 v，经 Levi-Civita transport 送到相邻 face v 上。
    实现方式：绕共享边方向，把法向 n_u 旋到 n_v，同时把向量一起旋过去。
    """
    theta = math.atan2(
        np.dot(edge_dir, np.cross(n_u, n_v)),
        np.dot(n_u, n_v)
    )
    return rotate_axis_angle(v, edge_dir, theta)


def build_face_local_frames(surface):
    """
    每个 face 构造一个局部正交基 (t1, t2, n)
    假设输入严格是无边界连通闭曲面，不做任何检查。
    """
    verts = surface.verts
    faces = surface.faces
    F = surface.F

    t1 = np.zeros((F, 3), dtype=np.float64)
    t2 = np.zeros((F, 3), dtype=np.float64)
    normals = np.zeros((F, 3), dtype=np.float64)

    for f_idx, (i, j, k) in enumerate(faces):
        n = surface.get_normal(i, j, k)
        normals[f_idx] = n

        e = verts[j] - verts[i]
        e = e - np.dot(e, n) * n
        t1[f_idx] = normalize(e)
        t2[f_idx] = np.cross(n, t1[f_idx])

    return t1, t2, normals


def build_vector_field_from_phi(
    surface,
    phi,
    seed_face=0,
    seed_angle=0.0,
    use_minus_phi=False,
):
    """
    从 dual 1-form phi 构造每个 face 上的单位切向量场。

    参数：
        surface: ClosedSurface
        phi: shape (surface.E,)
        seed_face: 起始 face
        seed_angle: 起始向量相对局部基 (t1, t2) 的角度
        use_minus_phi: 是否整体反号

    返回：
        face_vectors: (F, 3)
        face_centers: (F, 3)
        normals: (F, 3)
    """
    verts = surface.verts
    faces = surface.faces
    F = surface.F

    t1, t2, normals = build_face_local_frames(surface)

    face_vectors = np.zeros((F, 3), dtype=np.float64)
    visited = np.zeros(F, dtype=bool)

    face_vectors[seed_face] = (
        math.cos(seed_angle) * t1[seed_face]
        + math.sin(seed_angle) * t2[seed_face]
    )
    face_vectors[seed_face] = normalize(face_vectors[seed_face])

    visited[seed_face] = True
    q = deque([seed_face])

    while q:
        fu = q.popleft()
        vu = face_vectors[fu]
        nu = normals[fu]

        for fv in surface.co_adj_mat[fu]:
            if visited[fv]:
                continue

            dual_edge = (min(fu, fv), max(fu, fv))
            edge = surface.dual_edge_to_edge[dual_edge]
            e_idx = surface.dual_edge_to_index[dual_edge]

            i, j = edge
            edge_dir = normalize(verts[j] - verts[i])

            nv = normals[fv]

            # 1) Levi-Civita transport across primal edge
            vv = levi_civita_transport_across_edge(vu, nu, nv, edge_dir)

            # 2) 按 dual edge 的真实定向加上 phi
            left_face, right_face = surface.order_of_dual_edge(edge)
            theta = phi[e_idx] if left_face == fu else -phi[e_idx]

            if use_minus_phi:
                theta = -theta

            vv = rotate_axis_angle(vv, nv, theta)

            # 投回 fv 的切平面
            vv = vv - np.dot(vv, nv) * nv
            vv = normalize(vv)

            face_vectors[fv] = vv
            visited[fv] = True
            q.append(fv)

    face_centers = verts[faces].mean(axis=1)
    return face_vectors, face_centers, normals
