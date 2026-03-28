import numpy as np
import pyvista as pv
import os
import glob

NEURON_ID = "720575940568708429"
SWC_PATH = f"data/skeletons/{NEURON_ID}.swc"
MESH_DIR = "data/meshes"

# ── Load skeleton ──────────────────────────────────────────────────────────────
data = np.loadtxt(SWC_PATH, comments="#")
ids     = data[:, 0].astype(int)
xyz     = data[:, 2:5]
parents = data[:, 6].astype(int)

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

# ── Load mesh(es) ──────────────────────────────────────────────────────────────
# Supports .obj, .ply, .stl, .vtk, .vtp — extend the glob if needed
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
mesh_files = list(dict.fromkeys(mesh_files))  # deduplicate, preserve order

if not mesh_files:
    raise FileNotFoundError(
        f"No mesh files found in '{MESH_DIR}'. "
        "Supported formats: .obj .ply .stl .vtk .vtp"
    )

meshes = []
for path in mesh_files:
    print(f"Loading mesh: {path}")
    m = pv.read(path)
    print(f"  Mesh — points: {m.n_points}, cells: {m.n_cells}")

    # Mesh is in nm, skeleton is in µm → scale by 1/1000
    m.points /= 1000.0

    # Align centers axis by axis
    for axis in range(3):
        skel_center = (xyz[:, axis].min() + xyz[:, axis].max()) / 2
        mesh_center = (m.points[:, axis].min() + m.points[:, axis].max()) / 2
        m.points[:, axis] += skel_center - mesh_center

    meshes.append(m)

# ── Sanity-check bounding boxes ────────────────────────────────────────────────
skel_bounds = skeleton.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
print(f"\nSkeleton bounds: x=[{skel_bounds[0]:.1f}, {skel_bounds[1]:.1f}]"
      f"  y=[{skel_bounds[2]:.1f}, {skel_bounds[3]:.1f}]"
      f"  z=[{skel_bounds[4]:.1f}, {skel_bounds[5]:.1f}]")
for m, path in zip(meshes, mesh_files):
    b = m.bounds
    print(f"Mesh ({os.path.basename(path)}) bounds:"
          f" x=[{b[0]:.1f}, {b[1]:.1f}]"
          f" y=[{b[2]:.1f}, {b[3]:.1f}]"
          f" z=[{b[4]:.1f}, {b[5]:.1f}]")

# ── Render ─────────────────────────────────────────────────────────────────────
p = pv.Plotter()

# Mesh: translucent surface so the skeleton is visible through it
for m in meshes:
    p.add_mesh(
        m,
        color="steelblue",
        opacity=0.35,
        label="Mesh",
        show_edges=False,
    )

# Skeleton: bright, solid lines on top
p.add_mesh(
    skeleton,
    color="red",
    line_width=3,
    label="Skeleton",
)

p.add_legend()
p.show_bounds(grid="front", location="outer", all_edges=True)
p.show_axes()

p.enable_parallel_projection()
p.view_xy()
p.show()