# %%
from caveclient import CAVEclient

# -------------------------------------------------
# Config
# -------------------------------------------------
DATASET = "stroeh_mouse_retina"
AFTER_ROOT_ID = 720575940557940516

# -------------------------------------------------
# Initialize client
# -------------------------------------------------
client = CAVEclient(DATASET)

# -------------------------------------------------
# Get lineage graph and original IDs
# -------------------------------------------------
lg = client.chunkedgraph.get_lineage_graph(AFTER_ROOT_ID)

sources = {link["source"] for link in lg["links"]}
targets = {link["target"] for link in lg["links"]}
original_ids = sources - targets

print(f"Found {len(original_ids)} original IDs")

# -------------------------------------------------
# Get AFTER leaves once
# -------------------------------------------------
after_vox = set(client.chunkedgraph.get_leaves(root_id=AFTER_ROOT_ID))

# -------------------------------------------------
# Compute IoU, Precision, Recall vs each original ID
# -------------------------------------------------
best_iou = -1
best_id = None

results = []

for oid in original_ids:
    before_vox = set(client.chunkedgraph.get_leaves(root_id=oid))

    tp = len(before_vox & after_vox)
    fp = len(before_vox - after_vox)
    fn = len(after_vox - before_vox)

    union = tp + fp + fn

    iou = tp / union if union > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print(f"{oid}: IoU={iou:.6f}, Precision={precision:.6f}, Recall={recall:.6f}")

    results.append((oid, iou, precision, recall))

    if iou > best_iou:
        best_iou = iou
        best_id = oid

# -------------------------------------------------
# Report best match
# -------------------------------------------------
print("\nBest match (by IoU):")
for oid, iou, precision, recall in results:
    if oid == best_id:
        print(f"Original ID: {oid}")
        print(f"IoU: {iou:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")

# %%
