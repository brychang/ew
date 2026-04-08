# This script is intended to visualize the original and proofread meshes for a cell.
import pyvista as pv

# rgc
# A = pv.read("data/rgc_pfd/720575940573143897.obj")  # original mesh
# B = pv.read("data/rgc_pfd/720575940553314647.obj")  # proofread mesh

# nnos2
# A = pv.read("data/nnos2_pfd/720575940568640409.obj")
# B = pv.read("data/nnos2_pfd/720575940568708429.obj")

# t5o
# A = pv.read("data/t5o_pfd/720575940562643603.obj")
# B = pv.read("data/t5o_pfd/720575940562643603.obj")

# nnos2 03-25
A = pv.read("data/meshes_2026-03-25_14-45-12/720575940568658329.obj")
B = pv.read("data/meshes_2026-03-25_11-00-31/720575940585312278.obj")

# rgc 03-25
# A = pv.read("data/meshes_2026-03-25_14-45-12/720575940575255862.obj")
# B = pv.read("data/meshes_2026-03-25_11-00-31/720575940557127939.obj")

# bc 03-25
# A = pv.read("data/meshes_2026-03-25_11-00-31/720575940574792758.obj")
# B = pv.read("data/meshes_2026-03-25_11-00-31/720575940574792758.obj")

# sac 03-26
# A = pv.read("data/meshes_2026-03-26_12-33-46/720575940568751449.obj")
# B = pv.read("data/meshes_2026-03-26_12-33-46/720575940570797089.obj")

p = pv.Plotter()

# individual neurons
p.add_mesh(A, color="red", opacity=0.5)
# light green
p.add_mesh(B, color="lightgreen", opacity=0.5)


p.show()
