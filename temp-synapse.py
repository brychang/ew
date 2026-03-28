import time
from pathlib import Path

import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from precomputed_python import AnnotationWriter
from tqdm import tqdm

df = pd.read_csv("data/ribbon_v2_info.df", header=0, index_col=0)

# Filter for ribbon synapses with reasonable sizes
filtered_df = df[(df["size"] > 100) & (df["size"] < 1000)]

points = filtered_df[["centroid_x", "centroid_y", "centroid_z"]].values.astype(
    np.float32
)

# Set bounding box to a small region
p1 = 35848, 30328, 900
p2 = 55686, 45431, 1300
# Given p1 and p2 as the min and max of the points, we can set a bounding box around these values.
xmin, ymin, zmin = np.minimum(p1, p2)
xmax, ymax, zmax = np.maximum(p1, p2)

# Filter points within the bounding box
bbox_filtered_df = filtered_df[
    (filtered_df["centroid_x"] >= xmin)
    & (filtered_df["centroid_x"] <= xmax)
    & (filtered_df["centroid_y"] >= ymin)
    & (filtered_df["centroid_y"] <= ymax)
    & (filtered_df["centroid_z"] >= zmin)
    & (filtered_df["centroid_z"] <= zmax)
]

# Fetch the actual resolution from the precomputed image source
img_vol = CloudVolume("precomputed://gs://stroeh_sem_mouse_retina/image/v2")
resolution = img_vol.resolution.tolist()
voxel_size = np.array(resolution)

writer = AnnotationWriter(
    annotation_type="point",
    names=["x", "y", "z"],
    scales=resolution,
    units="nm",
)

# Add points to the annotation writer, converting from voxel coordinates to physical units
for i, row in tqdm(bbox_filtered_df.iterrows(), total=bbox_filtered_df.shape[0]):
    writer.add_point(
        point=[
            row.centroid_x,
            row.centroid_y,
            row.centroid_z,
        ],
        id=i,
    )
    print(f"Added point {i}: ({row.centroid_x}, {row.centroid_y}, {row.centroid_z})")

# Print number of original and filtered points, and the bounding box extent
print("Original:", df.shape[0])
print("Filtered:", filtered_df.shape[0])
print("Bounded:", bbox_filtered_df.shape[0])

print("BBox extent:")
print("x:", bbox_filtered_df["centroid_x"].min(), bbox_filtered_df["centroid_x"].max())
print("y:", bbox_filtered_df["centroid_y"].min(), bbox_filtered_df["centroid_y"].max())
print("z:", bbox_filtered_df["centroid_z"].min(), bbox_filtered_df["centroid_z"].max())

# Create output directory with timestamp
out = Path("data/annotations_ribbon_" + time.strftime("%Y-%m-%d_%H-%M-%S")).resolve()
out.mkdir(exist_ok=True)

# Write the annotations to disk
writer.write(out)
print(f"Annotations written to {out}")
