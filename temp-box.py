import numpy as np
import pyvista as pv
import os
import glob
from cloudvolume import CloudVolume

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
NEURON_ID = "720575940568708429"
SWC_PATH = f"data/skeletons/{NEURON_ID}.swc"
MESH_DIR = "data/meshes"

SCALE_BAR_UM = 150.0
SCALE_BAR_X_OFFSET = 120.0
SCALE_BAR_Y_OFFSET = 120.0
LABEL_Y_OFFSET = -60.0

Z_PLANE = 0.0   # Everything flattened here


# ─────────────────────────────────────────────────────────────
# Load Skeleton
# ─────────────────────────────────────────────────────────────
data = np.loadtxt(SWC_PATH, comments="#")

ids     = data[:,0].astype(int)
xyz     = data[:,2:5]
parents = data[:,6].astype(int)

# Flatten to 2D
xyz[:,2] = Z_PLANE

id_to_idx = {nid:i for i,nid in enumerate(ids)}

edges = []
for i,p in enumerate(parents):
    if p == -1:
        continue
    if p in id_to_idx:
        edges.append([i, id_to_idx[p]])

edges = np.array(edges, dtype=np.int64)

print(f"Skeleton — nodes: {len(xyz)}, edges: {len(edges)}")

lines = np.hstack([np.full((len(edges),1),2), edges]).ravel()

skeleton = pv.PolyData(xyz)
skeleton.lines = lines


# ─────────────────────────────────────────────────────────────
# Load Mesh (optional)
# ─────────────────────────────────────────────────────────────
mesh_patterns = [
    os.path.join(MESH_DIR, f"{NEURON_ID}.*"),
    os.path.join(MESH_DIR, "*.obj"),
    os.path.join(MESH_DIR, "*.ply"),
    os.path.join(MESH_DIR, "*.stl"),
    os.path.join(MESH_DIR, "*.vtk"),
    os.path.join(MESH_DIR, "*.vtp"),
]

mesh_files = []
for pattern in mesh_patterns:
    mesh_files.extend(glob.glob(pattern))

mesh_files = list(dict.fromkeys(mesh_files))

meshes = []

for path in mesh_files:
    print("Loading mesh:", path)

    m = pv.read(path)

    print(f"  Mesh — points: {m.n_points}, cells: {m.n_cells}")

    # nm → µm
    m.points /= 1000.0

    # flatten mesh
    m.points[:,2] = Z_PLANE

    meshes.append(m)


# ─────────────────────────────────────────────────────────────
# Get EM Volume Size (for bounding box)
# ─────────────────────────────────────────────────────────────
vol = CloudVolume(
    "gs://stroeh_sem_mouse_retina/image/v2/",
    mip=4,
    use_https=True
)

resolution_nm = np.array(vol.scale["resolution"])
bounds = vol.bounds

vol_size_vox = np.array(bounds.maxpt) - np.array(bounds.minpt)
vol_size_nm  = vol_size_vox * resolution_nm
vol_size_um  = vol_size_nm / 1000.0

x_size = vol_size_um[0]
y_size = vol_size_um[1]

print(f"EM bounding box: x=[0,{x_size:.1f}] µm  y=[0,{y_size:.1f}] µm")


# ─────────────────────────────────────────────────────────────
# Create Bounding Box
# ─────────────────────────────────────────────────────────────
box_corners = np.array([
    [0,      0,      Z_PLANE],
    [x_size, 0,      Z_PLANE],
    [x_size, y_size, Z_PLANE],
    [0,      y_size, Z_PLANE],
    [0,      0,      Z_PLANE]
])

bbox_poly = pv.lines_from_points(box_corners)


# ─────────────────────────────────────────────────────────────
# Scale Bar
# ─────────────────────────────────────────────────────────────
scale_bar_start = np.array([
    SCALE_BAR_X_OFFSET,
    SCALE_BAR_Y_OFFSET,
    Z_PLANE
])

scale_bar_end = scale_bar_start + np.array([SCALE_BAR_UM,0,0])

scale_bar = pv.Line(scale_bar_start, scale_bar_end)

label_pos = np.array([
    scale_bar_start[0] + SCALE_BAR_UM/2,
    scale_bar_start[1] + LABEL_Y_OFFSET,
    Z_PLANE
])


# ─────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────
p = pv.Plotter()

# Mesh
# for m in meshes:
#     p.add_mesh(
#         m,
#         color="steelblue",
#         opacity=0.3,
#         show_edges=False
#     )

# Skeleton
p.add_mesh(skeleton.extract_all_edges(), color="red", line_width=2)

# Bounding box
p.add_mesh(
    bbox_poly,
    color="steelblue",
    line_width=4
)

# Scale bar
p.add_mesh(
    scale_bar,
    color="black",
    line_width=8
)

# Label below bar
p.add_point_labels(
    [label_pos],
    ["150 µm"],
    font_size=20,
    text_color="black",
    point_size=0,
    shape=None
)

# True 2D camera
p.enable_parallel_projection()
p.view_xy()

p.show()