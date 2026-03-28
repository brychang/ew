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

# -------------------------------------------------
# 2. Define bounding box from e2198 (RGC-centered)
# -------------------------------------------------

# --- Load SWCs to compute centers (same as reference script)
NEURON_IDS = ["720575940585312278", "720575940557127939", "720575940574792758"]
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

# --- Construct e2198 box (350 µm × 350 µm)
box_um = np.array(
    [
        [rgc_center[0] - 175, rgc_center[1] - 175],
        [rgc_center[0] + 175, rgc_center[1] - 175],
        [rgc_center[0] + 175, rgc_center[1] + 175],
        [rgc_center[0] - 175, rgc_center[1] + 175],
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

# keep your original z range
z0 = 1
z1 = 2064

z_face = vol.download(Bbox((x0, y0, z1 - 1), (x1, y1, z1))).squeeze()
y_face = vol.download(Bbox((x0, y1 - 1, z0), (x1, y1, z1))).squeeze()
x_face = vol.download(Bbox((x1 - 1, y0, z0), (x1, y1, z1))).squeeze()


# -------------------------------------------------
# 3. Plane construction
# -------------------------------------------------
def make_plane(data, origin_voxel, normal_axis):
    nx, ny = data.shape

    if normal_axis == "z":
        dims = (nx, ny, 1)
    elif normal_axis == "y":
        dims = (nx, 1, ny)
    elif normal_axis == "x":
        dims = (1, nx, ny)
    else:
        raise ValueError("normal_axis must be x, y, or z")

    origin_physical = np.array(origin_voxel) * spacing

    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin_physical)

    grid.point_data["values"] = data.flatten(order="F")
    return grid


plane_z = make_plane(z_face, (x0, y0, z1 - 1), "z")
plane_y = make_plane(y_face, (x0, y1 - 1, z0), "y")
plane_x = make_plane(x_face, (x1 - 1, y0, z0), "x")

# -------------------------------------------------
# 4. Load meshes and keep IDs
# -------------------------------------------------
mesh_dir = Path("data/meshes_2026-03-25_11-00-31")

mesh_files = sorted(f for f in mesh_dir.glob("*.obj") if f.stem.isdigit())

mesh_ids = NEURON_IDS
meshes = [pv.read(f) for f in mesh_files]

# -------------------------------------------------
# 5. Rendering
# -------------------------------------------------
p = pv.Plotter()

# Background planes, with lighter colors
# Normalize once
for plane in [plane_z, plane_y, plane_x]:
    arr = plane["values"].astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = arr * 0.5 + 0.5  # very light
    plane["values"] = arr

# Render with fixed mapping
p.add_mesh(plane_z, cmap="gray", clim=[0, 1], show_scalar_bar=False)
p.add_mesh(plane_y, cmap="gray", clim=[0, 1], show_scalar_bar=False)
p.add_mesh(plane_x, cmap="gray", clim=[0, 1], show_scalar_bar=False)

# Add meshes and store actors
actors = []
colors = ["#0072B2", "#D55E00", "#009E73"]
for mesh, color in zip(meshes, colors):
    actor = p.add_mesh(mesh, color=color, opacity=0.5)
    actors.append(actor)


# -------------------------------------------------
# 6. Checkbox toggle logic
# -------------------------------------------------
def make_checkbox_callback(actor):
    def toggle(flag):
        actor.SetVisibility(flag)
        p.render()

    return toggle


start_y = 10
for i, (actor, mesh_id) in enumerate(zip(actors, mesh_ids)):
    p.add_checkbox_button_widget(
        callback=make_checkbox_callback(actor),
        value=True,
        position=(10, start_y + i * 35),
        size=25,
        border_size=1,
    )
    p.add_text(
        f"{mesh_id}",  # <-- use real mesh ID
        position=(45, start_y + i * 35),
        font_size=10,
    )


# -------------------------------------------------
# 6.5 Keypress: Randomize mesh colors (press "L")
# -------------------------------------------------
def randomize_colors():
    for actor in actors:
        rgb = np.random.rand(3)  # random RGB in [0,1]
        actor.GetProperty().SetColor(rgb)
    p.render()


p.add_key_event("l", randomize_colors)

# -------------------------------------------------
# 7. Visualization mode
# -------------------------------------------------
INTERACTIVE_MODE = False  # Set to True for interactive view, False for snapshots

if INTERACTIVE_MODE:
    # Interactive mode
    p.camera_position = "iso"
    p.show()
else:
    # Static orthographic snapshots
    Path("snapshots").mkdir(exist_ok=True)

    # Combine all meshes bounds + planes
    all_bounds = np.array(
        [plane_x.bounds, plane_y.bounds, plane_z.bounds]
        + [mesh.bounds for mesh in meshes]
    )
    xmin = all_bounds[:, 0].min()
    xmax = all_bounds[:, 1].max()
    ymin = all_bounds[:, 2].min()
    ymax = all_bounds[:, 3].max()
    zmin = all_bounds[:, 4].min()
    zmax = all_bounds[:, 5].max()
    bbox_center = [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2]
    bbox_size = max(xmax - xmin, ymax - ymin, zmax - zmin)

    # Use a higher-resolution off-screen plotter
    p = pv.Plotter(off_screen=True, window_size=(2048, 2048))

    # Add planes and meshes
    p.add_mesh(plane_z, cmap="gray", clim=[0, 1], show_scalar_bar=False)
    p.add_mesh(plane_y, cmap="gray", clim=[0, 1], show_scalar_bar=False)
    p.add_mesh(plane_x, cmap="gray", clim=[0, 1], show_scalar_bar=False)
    for mesh, color in zip(meshes, colors):
        actor = p.add_mesh(mesh, color=color, opacity=0.5)

    # Orthographic camera
    p.camera.parallel_projection = True

    directions = {
        "neg_x": ([-1, 0, 0], [0, 0, 1]),
        "neg_z": ([0, 0, -1], [0, 1, 0]),
    }

    distance = bbox_size * 1.2

    for name, (direction, view_up) in directions.items():
        pos = np.array(bbox_center) + np.array(direction) * distance
        p.camera_position = (pos.tolist(), bbox_center, view_up)
        p.render()
        d = time.strftime("%Y-%m-%d_%H-%M-%S")
        out = Path(f"snapshots/{name}_{d}.png")
        p.screenshot(f"snapshots/{name}_{d}.png", scale=4)

    p.close()
    print("Snapshots saved in ./snapshots/")
