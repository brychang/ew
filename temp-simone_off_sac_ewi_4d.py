# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import skeliner as sk
from scipy.spatial import cKDTree

# %%
df = pd.read_csv("data/OFF SAC_720575940567969903.csv", index_col=0)
# %%
# Get summary statistics of distance_nm column
print(df["distance_nm"].describe())


# %%
# Define colors for cell types
colors = {
    "t1": "#188a53",
    "t2": "#e7bd13",
    "t3a": "#1aade8",
    "t3b": "#045684",
    "t4": "#951f92",
    "t5t": "#e41a1c",
    "GluMI": "#ff7f00",
}


# Ribbons seem to contact at pointA with and with contact area contact_area_um2.
# Use skeliner to compute the distance from pointA to the SAC soma.
def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def build_skeleton(mesh_path: Path, skeleton_seg_id: int | None):
    log(f"Loading mesh: {mesh_path}")
    mesh = sk.io.load_mesh(str(mesh_path))
    log("Skeletonizing mesh")
    return sk.skeletonize(
        mesh,
        detect_soma=True,
        collapse_soma=True,
        bridge_gaps=True,
        prune_tiny_neurites=True,
        unit="nm",
        id=skeleton_seg_id,
        verbose=True,
    )


# %%

skeleton = build_skeleton(
    Path("data/meshes_2026-04-06_16-20-29/720575940567969903.obj"),
    skeleton_seg_id=720575940567969903,
)

# %%

# Get soma object
soma = skeleton.soma

# %%
# Soma has a function distance defined this way:
# def distance(self, x, to="center"):
#         """
#         Compute the distance from *x* to the soma.

#         Parameters
#         ----------
#         x : (N, 3) or (3,) array-like
#             Points in world coordinates.
#         to : {'center', 'surface'}
#             Whether to compute the distance to the center or to the surface.


#         Returns
#         -------
#         (N,) or float
#             Unsigned Euclidean distance from *x* to the soma.
#         """
# Use it to compute the distance from the pointA column to the soma center and surface and add a column in the dataframe called distance_to_soma_center and distance_to_soma_surface.
# pointA column is in format '[np.float64(37625.0), np.float64(32897.0), np.float64(1426.0)]'
def parse_pointA(pointA_str: str) -> list[float]:
    # Remove brackets and split by comma
    pointA_str = pointA_str.strip("[]")
    return [float(coord.split("(")[1].split(")")[0]) for coord in pointA_str.split(",")]


df["pointA_parsed"] = df["pointA"].apply(parse_pointA)
df["distance_to_soma_center"] = df["pointA_parsed"].apply(
    lambda point: soma.distance(point, to="center")
)
df["distance_to_soma_surface"] = df["pointA_parsed"].apply(
    lambda point: soma.distance(point, to="surface")
)

# %%
# Also add columns converting the distances from nm to µm.
df["distance_to_soma_center_um"] = df["distance_to_soma_center"] / 1000
df["distance_to_soma_surface_um"] = df["distance_to_soma_surface"] / 1000
# %%
# Plot a histogram where the x-axis is distance_to_soma_center_um and the y-axis is the count of contacts in that distance bin. Use bins of size 10 µm.
# Plot it as line plot with markers at the bin centers. Each cell type (column cell_type) should be a different line in the plot. Add a legend to indicate which line corresponds to which cell type.
plt.figure(figsize=(10, 6))
for cell_type, group in df.groupby("cell_type"):
    plt.hist(
        group["distance_to_soma_center_um"],
        bins=np.arange(0, df["distance_to_soma_center_um"].max() + 10, 10),
        alpha=0.5,
        label=cell_type,
    )
plt.xlabel("Distance to Soma Center (µm)")
plt.ylabel("Count of Contacts")
plt.title("Histogram of Distances from Contacts to Soma Center")
plt.legend()
plt.show()

# %%
# Let's do a diagnosis, as distance to soma center seems to be around 800 um for all contacts, which seems quite far. Let's see the scale of the soma center.
print("Soma center:", soma.center)
# %%
# Let's also see the scale of the pointA coordinates.
print("PointA coordinates:", df["pointA_parsed"].iloc[0])
# %%
# The pointA coordinates need to first be converted into nm from voxels. The spacing is 16 nm in x and y and 40 nm in z. So we need to multiply the pointA coordinates by the spacing to get them in nm.
spacing = np.array([16, 16, 40])  # nm
df["pointA_parsed_nm"] = df["pointA_parsed"].apply(
    lambda point: np.array(point) * spacing
)
# Now let's recompute the distances using the pointA_parsed_nm column.
df["distance_to_soma_center_nm"] = df["pointA_parsed_nm"].apply(
    lambda point: soma.distance(point, to="center")
)
df["distance_to_soma_surface_nm"] = df["pointA_parsed_nm"].apply(
    lambda point: soma.distance(point, to="surface")
)
# Now let's convert these distances to µm.
df["distance_to_soma_center_um"] = df["distance_to_soma_center_nm"] / 1000
df["distance_to_soma_surface_um"] = df["distance_to_soma_surface_nm"] / 1000
# Now let's plot the histogram again with the updated distances.
plt.figure(figsize=(10, 6))
for cell_type, group in df.groupby("cell_type"):
    plt.hist(
        group["distance_to_soma_center_um"],
        bins=np.arange(0, df["distance_to_soma_center_um"].max() + 10, 10),
        alpha=0.5,
        label=cell_type,
    )
plt.xlabel("Distance to Soma Center (µm)")
plt.ylabel("Count of Contacts")
plt.title("Histogram of Distances from Contacts to Soma Center (Updated)")
plt.legend()
plt.show()
# %%
# This is more expected. Now we just need to replace the histogram with a line plot with markers at the bin centers. Each cell type (column cell_type) should be a different line in the plot. Add a legend to indicate which line corresponds to which cell type.
plt.figure(figsize=(10, 6))
bins = np.arange(0, df["distance_to_soma_center_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in df.groupby("cell_type"):
    counts, _ = np.histogram(group["distance_to_soma_center_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("Distance from OFF SAC soma (µm)")
plt.ylabel("Ribbon synapse counts")
plt.legend()
plt.show()

# %%
# Another version omits GluMI and t5t, which were not in the original paper plot.
plt.figure(figsize=(10, 6))
bins = np.arange(0, df["distance_to_soma_center_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in df.groupby("cell_type"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    counts, _ = np.histogram(group["distance_to_soma_center_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("Distance from OFF SAC soma (µm)")
plt.ylabel("Ribbon synapse counts")
plt.legend()
plt.show()
# %%
# Another version uses contact_area_um2 as the y-axis instead of counts, and plots the total contact area in each distance bin for each cell type.
plt.figure(figsize=(10, 6))
bins = np.arange(0, df["distance_to_soma_center_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in df.groupby("cell_type"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    group["distance_bin"] = pd.cut(
        group["distance_to_soma_center_um"], bins=bins, labels=bin_centers
    )
    total_contact_area = group.groupby("distance_bin")["contact_area_um2"].sum()
    plt.plot(
        bin_centers,
        total_contact_area,
        marker="o",
        label=cell_type,
        color=colors.get(cell_type),
    )
plt.xlabel("Distance from OFF SAC soma (µm)")
plt.ylabel("Total ribbon synapse contact area (µm^2)")
plt.legend()
plt.show()
# %%
# Similarly, we can apply this to the ON SAC
on_df = pd.read_csv("data/ON SAC_720575940563685101.csv", index_col=0)
# get soma
on_skeleton = build_skeleton(
    Path("data/meshes_2026-04-06_17-00-03/720575940563685101.obj"),
    skeleton_seg_id=720575940563685101,
)
on_soma = on_skeleton.soma
on_df["pointA_parsed"] = on_df["pointA"].apply(parse_pointA)
on_df["pointA_parsed_nm"] = on_df["pointA_parsed"].apply(
    lambda point: np.array(point) * spacing
)
on_df["distance_to_soma_center_nm"] = on_df["pointA_parsed_nm"].apply(
    lambda point: on_soma.distance(point, to="center")
)
on_df["distance_to_soma_center_um"] = on_df["distance_to_soma_center_nm"] / 1000
on_df["distance_bin"] = pd.cut(
    on_df["distance_to_soma_center_um"], bins=bins, labels=bin_centers
)
plt.figure(figsize=(10, 6))
for cell_type, group in on_df.groupby("cell_type"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    total_contact_area = group.groupby("distance_bin")["contact_area_um2"].sum()
    plt.plot(
        bin_centers,
        total_contact_area,
        marker="o",
        label=cell_type,
        color=colors.get(cell_type),
    )
plt.xlabel("Distance from ON SAC soma (µm)")
plt.ylabel("Total ribbon synapse contact area (µm^2)")
plt.legend()
plt.show()
# count version
plt.figure(figsize=(10, 6))
for cell_type, group in on_df.groupby("cell_type"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    counts, _ = np.histogram(group["distance_to_soma_center_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("Distance from ON SAC soma (µm)")
plt.ylabel("Ribbon synapse counts")
plt.legend()
plt.show()
# %%
# I will do the same for this other df which was a different way to predict the contact pairs.
# It is also a csv with different columns
# We don't actually have pointA (where the synapse touches the SAC) but we have the ribbon synapse coordinates with centroid_x, centroid_y, centroid_z columns. So we can use those as the coordinates to compute the distance to the soma.
# We have distance_to_skeleton_nm which we want to filter for <200 nm to get the predicted contact pairs.
# Filter for match_found=True (it means a cell type was assigned to the BC)
# Use the column Cell Type (machine) for the cell type of the BC
# Then we can do the same distance binning and plotting as before.

# Compute skeleton for the OFF SAC
skeleton = build_skeleton(
    Path("data/meshes_2026-04-06_16-20-29/720575940567969903.obj"),
    skeleton_seg_id=720575940567969903,
)
soma = skeleton.soma

# Process dataframe
spacing = np.array([16, 16, 40])  # nm

pred_df = pd.read_csv(
    "data/ribbon_seg_id_runs/run_20260407_115639/ribbon_with_cell_labels.csv"
)
pred_df = pred_df[pred_df["match_found"]]
pred_df = pred_df[pred_df["distance_to_skeleton_nm"] < 200]
pred_df["centroid_parsed"] = pred_df.apply(
    lambda row: [row["centroid_x"], row["centroid_y"], row["centroid_z"]], axis=1
)
pred_df["centroid_parsed_nm"] = pred_df["centroid_parsed"].apply(
    lambda point: np.array(point) * spacing
)
pred_df["distance_to_soma_center_nm"] = pred_df["centroid_parsed_nm"].apply(
    lambda point: soma.distance(point, to="center")
)
pred_df["distance_to_soma_center_um"] = pred_df["distance_to_soma_center_nm"] / 1000
pred_df["distance_bin"] = pd.cut(
    pred_df["distance_to_soma_center_um"], bins=bins, labels=bin_centers
)

# %%
pred_df.head()

# %%
# Plot the predicted contacts
plt.figure(figsize=(10, 6))
bins = np.arange(0, pred_df["distance_to_soma_center_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in pred_df.groupby("Cell Type (machine)"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    counts, _ = np.histogram(group["distance_to_soma_center_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("Distance from OFF SAC soma (µm)")
plt.ylabel("Ribbon synapse counts [skeliner distance threshold < 200 nm]")
plt.legend()
plt.show()

# %%
# We will now compute the distance to soma slightly differently.
# flatone is a github repo that can flatten the SAC into 2D.
# The 2D skeleton can be used to compute a geodesic distance from the synapse to the soma along the surface of the SAC, which may be more biologically relevant than the Euclidean distance we computed before.
# I ran flatone on the SAC skeleton and got a .swc file with the 2D skeleton. I will load this skeleton and use it to compute the geodesic distance from the synapse to the soma.
# The path is at ~/flatone/output/720575940567969903/skeleton_warped.swc
# .npz is also available at data/flatone_sac/720575940567969903_skeleton_warped.npz

path_to_flatone_skeleton = Path(
    "~/flatone/output/720575940567969903/skeleton_warped.swc"
).expanduser()

# We will load the flatone skeleton and compute the geodesic distance from the synapse to the soma along the surface of the SAC.
data = np.loadtxt(path_to_flatone_skeleton, comments="#")
ids = data[:, 0].astype(int)
xyz = data[:, 2:5]
parents = data[:, 6].astype(int)
id_to_idx = {nid: i for i, nid in enumerate(ids)}
edges = []
for i, p in enumerate(parents):
    if p == -1:
        continue
    if p in id_to_idx:
        edges.append((i, id_to_idx[p]))
# Now we have the flatone skeleton in xyz and edges format.

# Let's check the root node of the flatone skeleton, which should correspond to the soma center.
root_node = np.where(parents == -1)[0][0]
print("Root node (soma center) coordinates:", xyz[root_node])
# %%
# Let's check the root node of the original skeleton to see if it corresponds to the same location.
print("Original skeleton soma center coordinates:", soma.center)
# %%
# Let's check the z range of the flatone skeleton to see if it is indeed flattened.
print("Flatone skeleton z range:", xyz[:, 2].min(), xyz[:, 2].max())
# %%
# The z range is very small, so it is indeed flattened. We can now compute the geodesic distance from the synapse to the soma along the surface of the SAC using this flatone skeleton. We can use Dijkstra's algorithm on the graph defined by the edges to compute the shortest path from each synapse to the root node (soma center) and get the geodesic distance.
G = nx.Graph()
for i, j in edges:
    dist = np.linalg.norm(xyz[i] - xyz[j])
    G.add_edge(i, j, weight=dist)
# Let's visualize the flatone skeleton to check it looks correct.
# Also plot the synapse points on top of it to see where they are located.
plt.figure(figsize=(8, 8))
plt.scatter(xyz[:, 0], xyz[:, 1], s=1)
plt.scatter(xyz[root_node, 0], xyz[root_node, 1], color="red", label="Soma center")
synapse_points_um = np.array(pred_df["centroid_parsed_nm"].tolist()) / 1000
plt.scatter(
    synapse_points_um[:, 0],
    synapse_points_um[:, 1],
    s=6,
    color="orange",
    alpha=0.5,
    label="Synapse points",
)

plt.xlabel("X (um)")
plt.ylabel("Y (um)")
plt.title("Flatone skeleton of OFF SAC")
plt.legend()
plt.axis("equal")
plt.show()
# Now we can compute the geodesic distance from each synapse to the root node. Note the flatone skeleton is in um, so we need to convert the synapse coordinates to um as well before computing the distance.
pred_df["centroid_parsed_um"] = pred_df["centroid_parsed_nm"].apply(
    lambda point: np.array(point) / 1000
)

tree = cKDTree(xyz)
# Precompute geodesic distance (in um) from soma/root to every skeleton node.
dist_to_root_um = nx.single_source_dijkstra_path_length(G, root_node, weight="weight")


def compute_geodesic_distance_um(point_um):
    _, idx = tree.query(point_um)
    return dist_to_root_um.get(idx, np.nan)


pred_df["distance_to_soma_geodesic_um"] = pred_df["centroid_parsed_um"].apply(
    compute_geodesic_distance_um
)
# Now we can plot the histogram of geodesic distances.
plt.figure(figsize=(10, 6))
bins = np.arange(0, pred_df["distance_to_soma_geodesic_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in pred_df.groupby("Cell Type (machine)"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    counts, _ = np.histogram(group["distance_to_soma_geodesic_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("Geodesic distance from OFF SAC soma (µm)")
plt.ylabel("Ribbon synapse counts [skeliner distance threshold < 200 nm]")
plt.legend()
plt.title("Geodesic distance from synapse to OFF SAC soma along the surface of the SAC")
plt.show()


# The previous was good but we don't actually want to use the z coordinate of the synapse since the flatone skeleton is flattened in z. We should only use the x and y coordinates of the synapse to compute the distance to the skeleton.
# We can modify the compute_geodesic_distance_um function to ignore the z coordinate when querying the tree and computing the distance.
def compute_geodesic_distance_um_ignore_z(point_um):
    point_2d = point_um[:2]  # Ignore z coordinate
    skeleton_2d = xyz[:, :2]  # Use only x and y coordinates of the skeleton
    tree_2d = cKDTree(skeleton_2d)
    _, idx = tree_2d.query(point_2d)
    return dist_to_root_um.get(idx, np.nan)


pred_df["distance_to_soma_geodesic_ignore_z_um"] = pred_df["centroid_parsed_um"].apply(
    compute_geodesic_distance_um_ignore_z
)
# Now we can plot the histogram of geodesic distances ignoring z.
plt.figure(figsize=(10, 6))
bins = np.arange(0, pred_df["distance_to_soma_geodesic_ignore_z_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in pred_df.groupby("Cell Type (machine)"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    counts, _ = np.histogram(group["distance_to_soma_geodesic_ignore_z_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("2D geodesic distance from OFF SAC soma (µm)")
plt.ylabel("Ribbon synapse counts [skeliner distance threshold < 200 nm]")
plt.legend()
plt.show()

# %%
# Let's repeat the same geodesic distance computation for the other method of predicting contact pairs, which was the one in the original df (not the pred_df). We will compute the geodesic distance from the pointB_parsed_um (the ribbon synapse coordinates) to the soma using the same flatone skeleton.
df["pointB_parsed"] = df["pointB"].apply(parse_pointA)
df["pointB_parsed_nm"] = df["pointB_parsed"].apply(
    lambda point: np.array(point) * spacing
)
df["pointB_parsed_um"] = df["pointB_parsed_nm"].apply(
    lambda point: np.array(point) / 1000
)
df["distance_to_soma_geodesic_ignore_z_um"] = df["pointB_parsed_um"].apply(
    compute_geodesic_distance_um_ignore_z
)
# Now we can plot the histogram of geodesic distances for this original df as well.
plt.figure(figsize=(10, 6))
bins = np.arange(0, df["distance_to_soma_geodesic_ignore_z_um"].max() + 10, 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
for cell_type, group in df.groupby("cell_type"):
    if cell_type in ["GluMI", "t5t"]:
        continue
    counts, _ = np.histogram(group["distance_to_soma_geodesic_ignore_z_um"], bins=bins)
    plt.plot(
        bin_centers, counts, marker="o", label=cell_type, color=colors.get(cell_type)
    )
plt.xlabel("2D geodesic distance from OFF SAC soma (µm)")
plt.ylabel("Ribbon synapse counts [Simone]")
plt.legend()
plt.show()
# %%
pred_df = pd.read_csv(
    "data/ribbon_seg_id_runs/run_20260408_110904/ribbon_with_cell_labels.csv"
)
# Let's compute the number of each cell type in the pred_df and print it out.
# First filter for match_found=True.
pred_df = pred_df[pred_df["match_found"]]
# Then get the value counts of the Cell Type (machine) column.
cell_type_counts = pred_df["Cell Type (machine)"].value_counts()
print("Cell type counts in pred_df:")
print(cell_type_counts)
# Let's also check Cell Type values in the pred_df.
cell_type_human_counts = pred_df["Cell Type"].value_counts()
print("Cell type counts (human labeled) in pred_df:")
print(cell_type_human_counts)
# Repeat but count unique cells rather than counts of synapses.
unique_cells = pred_df.groupby("Cell Type (machine)")["final_seg_id"].nunique()
print("Unique cell counts in pred_df:")
print(unique_cells)
# And with human labels.
unique_cells_human = pred_df.groupby("Cell Type")["final_seg_id"].nunique()
print("Unique cell counts (human labeled) in pred_df:")
print(unique_cells_human)
# %%
