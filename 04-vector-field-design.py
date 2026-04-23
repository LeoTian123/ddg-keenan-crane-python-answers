from closed_surface import *
from pyvista_wrapped import *

cs = ClosedSurface('input/double-torus.obj')

# for g=2
point_indices = [536, 368]
singularity_type_input = [-1, -1]

# for g=0
# point_indices = [536, 836, 1015]
# singularity_type_input = [2, 1, -1]

# for g=0
# point_indices = [224, 567]
# singularity_type_input = [1, 1]

# for g=1
# point_indices = []
# singularity_type_input = []

d0 = cs.d_0  # (|E| × |V|)
d1 = cs.d_1  # (|F| × |E|)
star0 = cs.star_0  # (|V| × |V|) diag
star1 = cs.star_1  # (|E| × |E|) diag
star2 = cs.star_2  # (|F| × |F|) diag
star0_inv = sp.diags(1.0 / star0.diagonal())
star1_inv = sp.diags(1.0 / star1.diagonal())
star2_inv = sp.diags(1.0 / star2.diagonal())

generators = cs.get_generators()
plot_cycles(cs.verts, cs.faces, generators)

gammas, omegas = cs.get_harmonic_bases(generators)
P = cs.get_P(generators, gammas)

print('About omega is closed: ')
for omega in omegas:
    print("||d omega|| =", np.linalg.norm(d1 @ omega))
print('About gamma is harmonic: ')
for gamma in gammas:
    print("||dγ|| =", np.linalg.norm(d1 @ gamma))
    print("||δγ|| =", np.linalg.norm(star0_inv @ d0.T @ star1 @ gamma))
print('Abour gammas are linearly independent: ')
print("2g =", len(generators))
if P.shape == (0, 0):
    print("rank(P) = 0")
else:
    print("rank(P) =", np.linalg.matrix_rank(P))

pyvista_multiple_edge_1_forms(cs.verts, cs.faces, cs.edges, gammas, factor=3)

phi, delta_beta = cs.get_trivial_connection(generators, gammas, point_indices, singularity_type_input)

# pyvista_edge_1_form(
#         cs.verts,
#         cs.faces,
#         cs.edges,
#         phi,
#         factor=0.3/1,
#         mesh_color="lightgray",
#         arrow_color="blue",
#         opacity=1.0,
#         show_edges=False,
#     )

pyvista_dual_edge_1_form(
    cs, phi,
    factor=0.5/2,
)

# The symbol of use_minus_phi is determined through trial and error,
# rather than manual inspection and analysis
face_vectors, face_centers, face_normals = build_vector_field_from_phi(
    cs, phi,
    seed_face=0,
    seed_angle=0.0,
    use_minus_phi=True,
)

pyvista_vectors_on_faces(
    verts=cs.verts,
    faces=cs.faces,
    face_vectors=face_vectors,
    face_centers=face_centers,
    vector_scale=0.4/10,
    show_surface=True,
    show_edges=False,
    surface_opacity=1,
    color="tomato",
    line_width=3.0,
    indices=point_indices,
)
