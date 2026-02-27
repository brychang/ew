from cloudvolume import CloudVolume, Bbox
import pyvista as pv


# -----------------------------
# 1. Connect to volume
# -----------------------------
vol = CloudVolume(
    "gs://stroeh_sem_mouse_retina/image/v2/",
    mip=4,
    use_https=True
)

# Bounding box
x0, y0, z0 = 1000, 1000, 500
x1, y1, z1 = 4000, 4000, 1000


# -----------------------------
# 2. Download ONLY back faces
# -----------------------------

# Back Z face (XY plane at z1)
z_face = vol.download(
    Bbox((x0, y0, z1 - 1), (x1, y1, z1))
).squeeze()

# Back Y face (XZ plane at y1)
y_face = vol.download(
    Bbox((x0, y1 - 1, z0), (x1, y1, z1))
).squeeze()

# Back X face (YZ plane at x1)
x_face = vol.download(
    Bbox((x1 - 1, y0, z0), (x1, y1, z1))
).squeeze()


# -----------------------------
# 3. Helper to create plane
# -----------------------------
def make_plane(data, origin, spacing=(1, 1, 1), normal_axis="z"):
    """
    Create a properly positioned PyVista plane
    from a 2D numpy array.
    """
    nx, ny = data.shape

    if normal_axis == "z":
        dims = (nx, ny, 1)
    elif normal_axis == "y":
        dims = (nx, 1, ny)
    elif normal_axis == "x":
        dims = (1, nx, ny)
    else:
        raise ValueError("normal_axis must be 'x', 'y', or 'z'")

    grid = pv.ImageData(
        dimensions=dims,
        spacing=spacing,
        origin=origin
    )

    grid.point_data["values"] = data.flatten(order="F")
    return grid


# -----------------------------
# 4. Create 3D planes
# -----------------------------

# Note: origin corresponds to voxel coordinates
plane_z = make_plane(
    z_face,
    origin=(x0, y0, z1),
    normal_axis="z"
)

plane_y = make_plane(
    y_face,
    origin=(x0, y1, z0),
    normal_axis="y"
)

plane_x = make_plane(
    x_face,
    origin=(x1, y0, z0),
    normal_axis="x"
)


# -----------------------------
# 5. Render
# -----------------------------
p = pv.Plotter()

p.add_mesh(plane_z, cmap="gray")
p.add_mesh(plane_y, cmap="gray")
p.add_mesh(plane_x, cmap="gray")

p.camera_position = "iso"
p.show()