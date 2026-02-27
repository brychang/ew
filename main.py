from cloudvolume import CloudVolume, Bbox
import pyvista as pv
import numpy as np

# 1. Connect to CloudVolume (mip 4 background)
vol = CloudVolume(
    "gs://stroeh_sem_mouse_retina/image/v2/",
    mip=4,
    use_https=True
)

# Physical voxel size at this mip (nanometers)
spacing = np.array(vol.scale["resolution"])

print("Using mip:", vol.mip)
print("Voxel spacing (nm):", spacing)

# 2. Define bounding box in voxels
x0, y0, z0 = 600, 600, 1
x1, y1, z1 = 2500, 3000, 2063

# 3. Download back faces of bounding box (voxel coords)
z_face = vol.download(
    Bbox((x0, y0, z1 - 1), (x1, y1, z1))
).squeeze()

y_face = vol.download(
    Bbox((x0, y1 - 1, z0), (x1, y1, z1))
).squeeze()

x_face = vol.download(
    Bbox((x1 - 1, y0, z0), (x1, y1, z1))
).squeeze()

# 4. Plane construction in physical space (nm)
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

    # Convert origin from voxels to physical nm
    origin_physical = np.array(origin_voxel) * spacing

    grid = pv.ImageData(
        dimensions=dims,
        spacing=spacing,
        origin=origin_physical
    )

    grid.point_data["values"] = data.flatten(order="F")
    return grid


plane_z = make_plane(z_face, (x0, y0, z1), "z")
plane_y = make_plane(y_face, (x0, y1, z0), "y")
plane_x = make_plane(x_face, (x1, y0, z0), "x")

# 5. Load meshes (nm)
mesh_ac  = pv.read("data/meshes/720575940563821025-ac.obj")  # amacrine cell
mesh_bc  = pv.read("data/meshes/720575940550272092-bc.obj")  # bipolar cell
mesh_rgc = pv.read("data/meshes/720575940577681609-rgc.obj") # retinal ganglion cell
mesh_sac = pv.read("data/meshes/720575940554910762-sac.obj") # starburst amacrine cell
print("Mesh bounds (nm):", mesh_ac.bounds)

# 6. Rendering
p = pv.Plotter()

# Add background planes
p.add_mesh(plane_z, cmap="gray")
p.add_mesh(plane_y, cmap="gray")
p.add_mesh(plane_x, cmap="gray")

# Add meshes as actors
actor_ac  = p.add_mesh(mesh_ac,  color="red",   opacity=0.4)
actor_bc  = p.add_mesh(mesh_bc,  color="blue",  opacity=0.4)
actor_rgc = p.add_mesh(mesh_rgc, color="green", opacity=0.4)
actor_sac = p.add_mesh(mesh_sac, color="orange", opacity=0.4)

# Start with only amacrine cell visible
actor_bc.SetVisibility(False)
actor_rgc.SetVisibility(False)
actor_sac.SetVisibility(False)

# 7. Mesh switching
def show_only(actor):
    actor_ac.SetVisibility(False)
    actor_bc.SetVisibility(False)
    actor_rgc.SetVisibility(False)
    actor_sac.SetVisibility(False)

    if actor is not None:
        actor.SetVisibility(True)

    p.render()

p.add_key_event("1", lambda: show_only(actor_ac))
p.add_key_event("2", lambda: show_only(actor_bc))
p.add_key_event("3", lambda: show_only(actor_rgc))
p.add_key_event("4", lambda: show_only(actor_sac))
p.add_key_event("0", lambda: show_only(None))

# 8. Final view (flipped 180 degrees)
p.camera_position = "iso"
p.show()