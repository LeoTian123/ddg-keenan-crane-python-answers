from basictools import *


def get_angle_defect(verts, adj_mat, p):
    res = 0
    for neighbor1 in adj_mat[p]:
        for neighbor2 in adj_mat[p]:
            if neighbor1 in adj_mat[neighbor2]:
                res += get_angle(verts, p, neighbor1, neighbor2)
    res = 4 * pi - res
    return res/2


# def get_angle_defects(verts, adj_mat):
#     angle_defects = np.empty(len(verts), dtype=np.float32)
#     for p in range(len(verts)):
#         angle_defects[p] = get_angle_defect(verts, adj_mat, p)
#     return angle_defects


def get_angle_defects(verts, faces):
    n = len(verts)
    angle_defects = np.full(n, 2 * pi, dtype=np.float32)

    for i, j, k in faces:
        angle_defects[i] -= get_angle(verts, i, j, k)
        angle_defects[j] -= get_angle(verts, j, k, i)
        angle_defects[k] -= get_angle(verts, k, i, j)

    return angle_defects


def get_Euler_Chi(verts, faces):
    angle_defects = get_angle_defects(verts, faces)
    return round(angle_defects.sum()/2/pi)


if __name__ == '__main__':
    verts, faces = resolve_input('input/bunny.obj')
    # angle_defects = get_angle_defects(verts, faces)
    print(get_Euler_Chi(verts, faces))