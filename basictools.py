import numpy as np
from collections import defaultdict
from copy import deepcopy
import math
pi = math.pi


def resolve_input(file_path):
    # def int_face(s):
    #     if '//' in s:
    #         return int(s.split('//')[0])
    #     else:
    #         return int(s)
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
    np_verts = np.array(verts, dtype=np.float32)
    np_faces = np.array(faces, dtype=np.int32) - 1
    return np_verts, np_faces


def get_angle(verts, p, a, b):
    p, a, b = verts[p], verts[a],verts[b]
    pa = a - p
    pb = b - p
    return math.acos(np.dot(pa,pb)/np.linalg.norm(pa)/np.linalg.norm(pb))


def get_normal(verts, v0, v1, v2):
    v0, v1, v2 = verts[v0], verts[v1], verts[v2]
    n = np.cross(v1 - v0, v2 - v0)
    n /= np.linalg.norm(n) + 1e-12
    return n


def get_area(verts, v0, v1, v2):
    v0, v1, v2 = verts[v0], verts[v1], verts[v2]
    n = np.cross(v1 - v0, v2 - v0)
    return np.linalg.norm(n)

def get_adj_mat(faces):
    adj_mat = defaultdict(set)

    for face in faces:
        v0, v1, v2 = face
        adj_mat[v0].add(v1)
        adj_mat[v1].add(v0)
        adj_mat[v1].add(v2)
        adj_mat[v2].add(v1)
        adj_mat[v2].add(v0)
        adj_mat[v0].add(v2)

    return adj_mat


def get_edge_to_opposite(faces):
    edge_to_opposite = defaultdict(list)

    for v0, v1, v2 in faces:
        v0, v1, v2 = sorted([v0, v1, v2])
        edge_to_opposite[(v0, v1)].append(v2)
        edge_to_opposite[(v0, v2)].append(v1)
        edge_to_opposite[(v1, v2)].append(v0)

    return edge_to_opposite


def get_link(adj_mat, p):
    res = []
    unordered_link = deepcopy(adj_mat[p])

    q = unordered_link.pop()
    res.append(q)

    # ... --- last --- q=first --- next=second --- ...
    # second, last = list(unordered_link & adj_mat[q])

    nxt = list(unordered_link & adj_mat[q])[0]
    res.append(nxt)
    try:
        while unordered_link:
            unordered_link.remove(nxt)
            if not unordered_link:
                break
            nxt = list(unordered_link & adj_mat[nxt])[0]
            res.append(nxt)
        return res
    except Exception as e:
        return res
