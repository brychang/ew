import matplotlib.pyplot as plt
import numpy as np
from cloudvolume import CloudVolume

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
# NEURON_IDS = ["720575940568708429", "720575940553314647", "720575940562643603"]
# NEURON_IDS = ["720575940585312278", "720575940557127939", "720575940574792758"]
NEURON_IDS = [720575940552419601, 720575940557940516, 720575940573955417]
SWC_PATHS = [f"data/skeletons/{id}.swc" for id in NEURON_IDS]

SCALE_BAR_UM = 150.0
SCALE_BAR_X_OFFSET = 120.0
SCALE_BAR_Y_OFFSET = 120.0
LABEL_Y_OFFSET = -60.0

Z_PLANE = 0.0


# ─────────────────────────────────────────────────────────────
# Load Skeletons
# ─────────────────────────────────────────────────────────────
skeletons = []

for swc_path in SWC_PATHS:
    data = np.loadtxt(swc_path, comments="#")

    ids = data[:, 0].astype(int)
    xyz = data[:, 2:5]
    parents = data[:, 6].astype(int)

    xyz[:, 2] = Z_PLANE  # flatten

    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    edges = []
    for i, p in enumerate(parents):
        if p == -1:
            continue
        if p in id_to_idx:
            edges.append((i, id_to_idx[p]))

    skeletons.append((xyz, edges))


# ─────────────────────────────────────────────────────────────
# Volume Bounding Box
# ─────────────────────────────────────────────────────────────
vol = CloudVolume("gs://stroeh_sem_mouse_retina/image/v2/", mip=4, use_https=True)

resolution_nm = np.array(vol.scale["resolution"])
bounds = vol.bounds

vol_size_vox = np.array(bounds.maxpt) - np.array(bounds.minpt)
vol_size_um = (vol_size_vox * resolution_nm) / 1000.0

x_size, y_size = vol_size_um[:2]


# ─────────────────────────────────────────────────────────────
# Compute centers
# ─────────────────────────────────────────────────────────────
centers = [xyz.mean(axis=0) for xyz, _ in skeletons]

bipolar_center = centers[2]
rgc_center = centers[1]


# ─────────────────────────────────────────────────────────────
# Helper: make rectangle
# ─────────────────────────────────────────────────────────────
def make_box(cx, cy, w, h):
    return np.array(
        [
            [cx - w / 2, cy - h / 2],
            [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2],
            [cx - w / 2, cy + h / 2],
            [cx - w / 2, cy - h / 2],
        ]
    )


# Full EM box
full_box = make_box(x_size / 2, y_size / 2, x_size, y_size)

# e2006
e2006_box = make_box(bipolar_center[0], bipolar_center[1], 114, 80)

# e2198
e2198_box = make_box(rgc_center[0], rgc_center[1], 350, 350)


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))

colors = ["#0072B2", "#D55E00", "#009E73"]

# Plot skeletons
for (xyz, edges), color in zip(skeletons, colors):
    for i, j in edges:
        x = [xyz[i, 0], xyz[j, 0]]
        y = [xyz[i, 1], xyz[j, 1]]
        ax.plot(x, y, color=color, linewidth=1)

# Bounding boxes
ax.plot(full_box[:, 0], full_box[:, 1], color="steelblue", linewidth=2)
ax.plot(e2006_box[:, 0], e2006_box[:, 1], color="gray", linewidth=2)
ax.plot(e2198_box[:, 0], e2198_box[:, 1], color="cyan", linewidth=2)

# Scale bar
margin = 300  # µm padding from edges

x0 = x_size - SCALE_BAR_UM - margin
y0 = y_size - margin

ax.plot([x0, x0 + SCALE_BAR_UM], [y0, y0], color="black", linewidth=4)

# Formatting
ax.set_aspect("equal")
ax.set_xlim(0, x_size)
ax.set_ylim(0, y_size)
ax.axis("off")

plt.tight_layout()
plt.show()
