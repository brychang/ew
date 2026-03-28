import time
from pathlib import Path

from cloudvolume import CloudVolume

# -------------------------------------------------
# Configuration
# -------------------------------------------------
# EXTRAS = list(original_ids)
EXTRAS = [720575940572159335]

CV_PATH = "graphene://middleauth+https://minnie.microns-daf.com/segmentation/table/stroeh_mouse_retina"
d = time.strftime("%Y-%m-%d_%H-%M-%S")
out = Path(f"data/meshes_{d}")
OUTPUT_DIR = out

# -------------------------------------------------
# Setup
# -------------------------------------------------
cv = CloudVolume(CV_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Download meshes
# -------------------------------------------------
extras = cv.mesh.get(EXTRAS, fuse=False)

for segid, mesh in extras.items():
    obj_str = mesh.to_obj()

    with open(OUTPUT_DIR / f"{segid}.obj", "wb") as f:
        f.write(obj_str)
