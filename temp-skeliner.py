# %%
# This is a temporary file to test skeliner.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skeliner as sk
from cloudvolume import CloudVolume
from tqdm import tqdm

# %%
# I want to test the distance function on a skeleton. I will use the skeleton of a neuron that I have in my data. I will load the skeleton using skeliner and then compute the distance from a test point to the skeleton.
# Load the skeleton
seg_id = 720575940572159335
MESH_PATH = "data/meshes_2026-03-27_16-02-07/720575940572159335.obj"
mesh = sk.io.load_mesh(MESH_PATH)
skel = sk.skeletonize(
    mesh,
    detect_soma=True,
    # --- post-processing parameters (all are defaults) ---
    collapse_soma=True,
    bridge_gaps=True,
    prune_tiny_neurites=True,
    # --- meta data ---
    unit="nm",  # mesh unit
    id=seg_id,
    # --- Optional ---
    verbose=True,
)
# %%
# Define a test point
test_point = np.array([64088, 46542, 1312]) * np.array([16, 16, 40])
# Compute the distance from the test point to the skeleton
distance = skel.distance(test_point)
print(f"Distance from test point to skeleton: {distance}")
# %%
# The distance function seems to be working. I will now load the centroids of the ribbon synapses.
df = pd.read_csv("data/ribbon_v2_info.df", header=0, index_col=0)

# Filter for ribbon synapses with reasonable sizes
filtered_df = df[(df["size"] > 100) & (df["size"] < 1000)]
points = filtered_df[["centroid_x", "centroid_y", "centroid_z"]].values.astype(
    np.float32
)
# %%
# I will now compute the distance from each of the ribbon synapse centroids to the skeleton and add it as a new column in the dataframe.
# I will use tqdm to show progress since this might take a while.
distances = []
for i, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    point = np.array([row.centroid_x, row.centroid_y, row.centroid_z]) * np.array(
        [16, 16, 40]
    )
    distance = skel.distance(point)
    distances.append(distance)
filtered_df["distance_to_skeleton"] = distances
# %%
# <ipython-input-23-4674d4b4cee8>:11: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead

# See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
#   filtered_df["distance_to_skeleton"] = distances
# This warning is because I am trying to set a new column on a filtered dataframe. To avoid this warning, I will create a copy of the filtered dataframe before setting the new column.
filtered_df = filtered_df.copy()
filtered_df["distance_to_skeleton"] = distances
# %%
# Now I can analyze the distances. I will first plot a histogram of the distances to see the distribution. I will first consider a range of 100-200 nm, which is a reasonable distance for ribbon synapses to be from the skeleton.
plt.hist(filtered_df["distance_to_skeleton"], bins=50, range=(100, 200))
plt.xlabel("Distance to Skeleton (nm)")
plt.ylabel("Count")
plt.title("Histogram of Distances from Ribbon Synapses to Skeleton")
plt.show()

# %%
# Let's take a detour and check that given a ribbon synapse centroid, I can get the cell ID of the cell that it beloongs to. I will use the cloudvolume library to fetch the segmentation ID at the location of the ribbon synapse centroid.
# Initialize the cloudvolume
vol = CloudVolume(
    "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina",
    mip=0,
    use_https=True,
)
# Get the segmentation ID at the location of the first ribbon synapse centroid
centroid = filtered_df.iloc[0][
    ["centroid_x", "centroid_y", "centroid_z"]
].values.astype(np.float32)
seg_id = vol[tuple(centroid.astype(int))]
print(f"Segmentation ID at the location of the first ribbon synapse centroid: {seg_id}")
# Also print the ribbon synapse ID for reference
ribbon_id = filtered_df.index[0]
print(f"Ribbon synapse ID: {ribbon_id}")
# And coordinate for reference
print(f"Centroid coordinates: {centroid}")
# %%
# I have confirmed that I can get the segmentation ID at the location of the ribbon synapse centroid. For all the ribbon synapses, I will get the segmentation ID and add it as a new column in the dataframe.
seg_ids = []
for i, row in tqdm(filtered_df.iterrows(), total=len(filtered_df)):
    centroid = row[["centroid_x", "centroid_y", "centroid_z"]].values.astype(np.float32)
    seg_id = vol[tuple(centroid.astype(int))]
    seg_ids.append(seg_id)
# %%
# This would take 300 hours if I do it sequentially. I will write a script to do this in parallel on a cluster.
# I will write the script below, but I will not run it here since it would take a long time. I will run it on a cluster with more resources.
# I should be splitting the dataframe into chunks and then running the above code on each chunk in parallel. I will use the multiprocessing library to do this.
# ---
# import multiprocessing as mp
# def get_seg_id(row):
#     centroid = row[["centroid_x", "centroid_y", "centroid_z"]].values.astype(np.float32)
#     seg_id = vol[tuple(centroid.astype(int))]
#     return seg_id
# if __name__ == "__main__":
#     with mp.Pool(processes=4) as pool:
#         seg_ids = pool.map(get_seg_id, [row for _, row in filtered_df.iterrows()])
# ---


# %%
# Make a copy of the filtered dataframe before setting the new column to avoid the SettingWithCopyWarning
filtered_df = filtered_df.copy()
filtered_df["segmentation_id"] = seg_ids
