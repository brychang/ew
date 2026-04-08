import io
import zipfile
from pathlib import Path

import requests

# %%
segid = 720575940573955417
url = f"https://codex-mouse.pniapps.org/ew2/download_skeleton_assets?segid={segid}"

# output paths
out_dir = Path("data/skeletons")
out_dir.mkdir(parents=True, exist_ok=True)
out_swc = out_dir / f"{segid}.swc"

# download zip into memory
response = requests.get(url)
response.raise_for_status()

with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    # find the SWC file
    swc_name = next(
        name for name in z.namelist() if name.endswith("skeleton_warped_morphopy.swc")
    )

    # extract and save with new name
    with z.open(swc_name) as src, open(out_swc, "wb") as dst:
        dst.write(src.read())

print(f"Saved SWC to {out_swc}")
