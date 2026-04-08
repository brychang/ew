# %%
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from cloudvolume import Bbox, CloudVolume

# -------------------------------------------------
# 1. Connect to CloudVolume
# -------------------------------------------------
vol = CloudVolume("gs://stroeh_sem_mouse_retina/image/v2/", mip=4, use_https=True)

spacing = np.array(vol.scale["resolution"])

# %%

# -------------------------------------------------
# 2. Define bounding box from e2198 (RGC-centered)
# -------------------------------------------------

# --- Load SWCs to compute centers (same as reference script)
NEURON_IDS = [720575940557940516, 720575940552419601, 720575940573955417]
SWC_PATHS = [f"data/skeletons/{id}.swc" for id in NEURON_IDS]

skeletons = []
for swc_path in SWC_PATHS:
    data = np.loadtxt(swc_path, comments="#")
    xyz = data[:, 2:5]
    skeletons.append(xyz)

# Compute centers
centers = [xyz.mean(axis=0) for xyz in skeletons]

# Match reference script: RGC = index 1
rgc_center = centers[1]
# %%
rgc_center
# %%
xyz
# %%
# Compute centers
centers = [xyz.mean(axis=0) for xyz in skeletons]


field_center = [201, 271]

# --- Construct e2198 box (350 µm × 350 µm)
box_um = np.array(
    [
        [field_center[0] - 175, field_center[1] - 175],
        [field_center[0] + 175, field_center[1] - 175],
        [field_center[0] + 175, field_center[1] + 175],
        [field_center[0] - 175, field_center[1] + 175],
    ]
)

# --- Convert µm → voxels
box_vox = box_um * 1000 / spacing[:2]

xmin, ymin = box_vox.min(axis=0)
xmax, ymax = box_vox.max(axis=0)

x0 = int(np.floor(xmin))
x1 = int(np.ceil(xmax))
y0 = int(np.floor(ymin))
y1 = int(np.ceil(ymax))
# %%
box_vox
# %%
x0, x1, y0, y1
# %%
import time
from pathlib import Path

name = "mesh_comparison_720575940557127939"

d = time.strftime("%Y-%m-%d_%H-%M-%S")
out = Path(f"snapshots/{name}_{d}.png")
out
# %%

# python temp-ribbon_seg_id.py --mesh-path data/meshes_2026-04-06_16-20-29/720575940567969903.obj --skeleton-seg-id 720575940567969903 --max-distance-nm 200 --workers 8 --chunk-size 2000 --format csv
