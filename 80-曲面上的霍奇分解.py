from operators import *
import scipy.sparse.linalg as spla

verts, faces = resolve_input('input/bunny.obj')
adj_mat = get_adj_mat(faces)

edge_to_opposite = get_edge_to_opposite(faces)
edges = list(edge_to_opposite.keys())

V, E, F = len(verts), len(edges), len(faces)
print(V - E + F)


d0 = get_d_0(E, V, edges) # (|E| × |V|)
d1 = get_d_1(F, E, faces, edges) # (|F| × |E|)

print(sp.linalg.norm(d1 @ d0))

star0 = get_hodge_star_0(verts, adj_mat)  # (|V| × |V|) diag
star1 = get_hodge_star_1(verts, edges, edge_to_opposite)  # (|E| × |E|) diag
star2 = get_hodge_star_2(verts, faces)  # (|F| × |F|) diag
star0_inv = sp.diags(1.0 / star0.diagonal())
star1_inv = sp.diags(1.0 / star1.diagonal())
star2_inv = sp.diags(1.0 / star2.diagonal())

K_dec = - d0.T @ star1 @ d0
L = get_laplacian_matrix(verts, adj_mat)

f = np.random.randn(V)
print("compare K:", np.linalg.norm(K_dec @ f - L @ f))
print("compare K:", sp.linalg.norm(K_dec - L))
print("compare star0_inv @ K_dec:", np.linalg.norm((star0_inv @ K_dec) @ f - L @ f))
print("compare star0_inv @ K_dec:", sp.linalg.norm(star0_inv @ K_dec - L))
print(K_dec[1, 1], L[1, 1])


omega = np.random.randn(E)


A = d0.T @ star1 @ d0          # (|V| × |V|)
b = d0.T @ star1 @ omega       # (|V|)
alpha = spla.spsolve(A, b)


A = d1 @ star1_inv @ d1.T     # (|F| × |F|)
b = d1 @ omega                # (|F|)
beta_tilde = spla.spsolve(A, b)
beta = star2_inv @ beta_tilde


grad_part = d0 @ alpha
curl_part = star1_inv @ d1.T @ star2 @ beta
gamma = omega - grad_part - curl_part


print("||γ|| =", np.linalg.norm(gamma))
print("||dγ|| =", np.linalg.norm(d1 @ gamma))
print("||δγ|| =", np.linalg.norm(star0_inv @ d0.T @ star1 @ gamma))
print("⟨dα, δβ⟩ =", grad_part.T @ star1 @ curl_part)
print("⟨dα, γ⟩ =", grad_part.T @ star1 @ gamma)
print("⟨δβ, γ⟩ =", curl_part.T @ star1 @ gamma)

