import numpy as np
import pyvista as pv
from cloudvolume import CloudVolume

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
NEURON_IDS = ["720575940568708429", "720575940553314647", "720575940562643603"]
SWC_PATHS = [f"data/skeletons/{id}.swc" for id in NEURON_IDS]
MESH_DIR = "data/meshes"

SCALE_BAR_UM = 150.0
SCALE_BAR_X_OFFSET = 120.0
SCALE_BAR_Y_OFFSET = 120.0
LABEL_Y_OFFSET = -60.0

Z_PLANE = 0.0  # Everything flattened here


# ─────────────────────────────────────────────────────────────
# Load Skeleton
# ─────────────────────────────────────────────────────────────
skeletons = []
for swc_path in SWC_PATHS:
    data = np.loadtxt(swc_path, comments="#")

    ids = data[:, 0].astype(int)
    xyz = data[:, 2:5]
    parents = data[:, 6].astype(int)

    # Flatten to 2D
    xyz[:, 2] = Z_PLANE

    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    edges = []
    for i, p in enumerate(parents):
        if p == -1:
            continue
        if p in id_to_idx:
            edges.append([i, id_to_idx[p]])

    edges = np.array(edges, dtype=np.int64)

    print(f"Skeleton — nodes: {len(xyz)}, edges: {len(edges)}")

    lines = np.hstack([np.full((len(edges), 1), 2), edges]).ravel()

    skeleton = pv.PolyData(xyz)
    skeleton.lines = lines
    skeletons.append(skeleton)

# ─────────────────────────────────────────────────────────────
# Get EM Volume Size (for bounding box)
# ─────────────────────────────────────────────────────────────
vol = CloudVolume("gs://stroeh_sem_mouse_retina/image/v2/", mip=4, use_https=True)

resolution_nm = np.array(vol.scale["resolution"])
bounds = vol.bounds

vol_size_vox = np.array(bounds.maxpt) - np.array(bounds.minpt)
vol_size_nm = vol_size_vox * resolution_nm
vol_size_um = vol_size_nm / 1000.0

x_size = vol_size_um[0]
y_size = vol_size_um[1]

print(f"EM bounding box: x=[0,{x_size:.1f}] µm  y=[0,{y_size:.1f}] µm")


# ─────────────────────────────────────────────────────────────
# Create Bounding Box (EWII)
# ─────────────────────────────────────────────────────────────
box_corners = np.array(
    [
        [0, 0, Z_PLANE],
        [x_size, 0, Z_PLANE],
        [x_size, y_size, Z_PLANE],
        [0, y_size, Z_PLANE],
        [0, 0, Z_PLANE],
    ]
)

bbox_poly = pv.lines_from_points(box_corners)

# ─────────────────────────────────────────────────────────────
# Compute neuron centers
# ─────────────────────────────────────────────────────────────
centers = [sk.points.mean(axis=0) for sk in skeletons]

bipolar_center = centers[2]  # 720575940562643603
rgc_center = centers[1]  # 720575940553314647


# ─────────────────────────────────────────────────────────────
# Bounding Box (e2006)
# 114 × 80 µm centered at bipolar cell
# ─────────────────────────────────────────────────────────────
w = 114
h = 80

cx, cy = bipolar_center[:2]

e2006_box_corners = np.array(
    [
        [cx - w / 2, cy - h / 2, Z_PLANE],
        [cx + w / 2, cy - h / 2, Z_PLANE],
        [cx + w / 2, cy + h / 2, Z_PLANE],
        [cx - w / 2, cy + h / 2, Z_PLANE],
        [cx - w / 2, cy - h / 2, Z_PLANE],
    ]
)

e2006_bbox_poly = pv.lines_from_points(e2006_box_corners)


# ─────────────────────────────────────────────────────────────
# Bounding Box (e2198)
# 350 × 350 µm centered at RGC
# ─────────────────────────────────────────────────────────────
w = 350
h = 350

cx, cy = rgc_center[:2]

e2198_box_corners = np.array(
    [
        [cx - w / 2, cy - h / 2, Z_PLANE],
        [cx + w / 2, cy - h / 2, Z_PLANE],
        [cx + w / 2, cy + h / 2, Z_PLANE],
        [cx - w / 2, cy + h / 2, Z_PLANE],
        [cx - w / 2, cy - h / 2, Z_PLANE],
    ]
)

e2198_bbox_poly = pv.lines_from_points(e2198_box_corners)
print(f"e2198 box corners (µm):\n{e2198_box_corners[:, :2]}")

# ─────────────────────────────────────────────────────────────
# Scale Bar
# ─────────────────────────────────────────────────────────────
scale_bar_start = np.array([SCALE_BAR_X_OFFSET, SCALE_BAR_Y_OFFSET, Z_PLANE])

scale_bar_end = scale_bar_start + np.array([SCALE_BAR_UM, 0, 0])

scale_bar = pv.Line(scale_bar_start, scale_bar_end)

label_pos = np.array(
    [
        scale_bar_start[0] + SCALE_BAR_UM / 2,
        scale_bar_start[1] + LABEL_Y_OFFSET,
        Z_PLANE,
    ]
)


# ─────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────
p = pv.Plotter()

# Skeleton
colors = ["red", "blue", "green", "orange", "purple"]
for i, skeleton in enumerate(skeletons):
    p.add_mesh(
        skeleton.extract_all_edges(), color=colors[i % len(colors)], line_width=2
    )

# Bounding boxes
p.add_mesh(bbox_poly, color="steelblue", line_width=4)
p.add_mesh(e2006_bbox_poly, color="gray", line_width=4)
p.add_mesh(e2198_bbox_poly, color="cyan", line_width=4)

# Scale bar
p.add_mesh(scale_bar, color="black", line_width=8)

# Label below bar
p.add_point_labels(
    [label_pos], ["150 µm"], font_size=20, text_color="black", point_size=0, shape=None
)

# True 2D camera
p.enable_parallel_projection()
p.view_xy()

p.show()
