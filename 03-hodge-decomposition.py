from closed_surface import *
from pyvista_wrapped import *

cs = ClosedSurface('input/double-torus.obj')

d0 = cs.d_0  # (|E| × |V|)
d1 = cs.d_1  # (|F| × |E|)
star0 = cs.star_0  # (|V| × |V|) diag
star1 = cs.star_1  # (|E| × |E|) diag
star2 = cs.star_2  # (|F| × |F|) diag
star0_inv = sp.diags(1.0 / star0.diagonal())
star1_inv = sp.diags(1.0 / star1.diagonal())
star2_inv = sp.diags(1.0 / star2.diagonal())

print("||d^2|| =", sp.linalg.norm(d1 @ d0))

L_by_operators = - d0.T @ star1 @ d0
L = cs.laplacian_matrix

print("compare d0.T @ star1 @ d0:", sp.linalg.norm(L_by_operators - L))
print("compare star0_inv @ d0.T @ star1 @ d0:", sp.linalg.norm(star0_inv @ L_by_operators - L))


omega = np.random.randn(cs.E)

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

form_1_list = [omega, grad_part, curl_part, gamma]

pyvista_multiple_edge_1_forms(cs.verts, cs.faces, cs.edges, form_1_list)