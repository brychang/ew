# %%
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from caveclient import CAVEclient

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DATASET = "stroeh_mouse_retina"
SEG_ID = 720575940559347573

# -------------------------------------------------
# Fetch lineage graph
# -------------------------------------------------
client = CAVEclient(DATASET)
# set timestamp_future to SEG_ID's timestamp.
lg = client.chunkedgraph.get_lineage_graph(SEG_ID)

# -------------------------------------------------
# Build NetworkX graph
# -------------------------------------------------
G = nx.DiGraph()

# Add nodes
for node in lg["nodes"]:
    G.add_node(node["id"], timestamp=node.get("timestamp"))

# Add edges
for link in lg["links"]:
    G.add_edge(link["source"], link["target"])

# -------------------------------------------------
# Plot graph
# -------------------------------------------------
plt.figure(figsize=(8, 6))

pos = nx.spring_layout(G, seed=0)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=800,
    font_size=8,
    arrows=True,
)

plt.title("ChunkedGraph Lineage")
plt.show()

# -------------------------------------------------
# Compute original IDs
# -------------------------------------------------
sources = {link["source"] for link in lg["links"]}
targets = {link["target"] for link in lg["links"]}

original_ids = sources - targets

# Kept for optional lineage debugging.

# %%
rid = 720575940559347573

lg = client.chunkedgraph.get_lineage_graph(
    rid,
    timestamp_past=client.chunkedgraph.get_root_timestamps(rid)[0],
    exclude_links_to_past=True,
)
print(lg)
tcl = client.chunkedgraph.get_tabular_change_log([rid])[rid]  # has user_id/user_name
print(tcl)
# %%
# get all nodes in lineage graph
sources = {link["source"] for link in lg["links"]}
targets = {link["target"] for link in lg["links"]}
all_rids = list(sources.union(targets))

# query change logs for all of them
tcl_all = client.chunkedgraph.get_tabular_change_log(all_rids)

# check which ones actually have entries
{k: v for k, v in tcl_all.items() if len(v) > 0}


# %%
# Search for rid within after_root_ids column, which is a list of ints.
def rid_in_after_root_ids(row, rid):
    after_root_ids = row.get("after_root_ids", [])
    return rid in after_root_ids


tcl_all[rid][tcl_all[rid].apply(lambda row: rid_in_after_root_ids(row, rid), axis=1)]

# %%
# Keep one operation per unique user: the one at that user's largest timestamp.
ts_unit = "ms"
tcl = tcl.copy()
tcl["ts_utc"] = pd.to_datetime(tcl["timestamp"], unit=ts_unit, utc=True)
latest_per_user = (
    tcl.sort_values(["user_id", "timestamp", "operation_id"])
    .groupby("user_id", as_index=False)
    .tail(1)
    .sort_values("timestamp")
)

result = latest_per_user[
    ["operation_id", "user_id", "user_name", "ts_utc", "after_root_ids"]
]
print(result.to_string(index=False))

# %%
ts_oldest = client.chunkedgraph.get_oldest_timestamp()
for oid in original_ids:
    ts = client.chunkedgraph.get_root_timestamps(oid)[0]
    print(f"Original ID {oid} has timestamp {ts} (oldest: {ts_oldest})")
    lg = client.chunkedgraph.get_lineage_graph(
        oid,
        timestamp_future=client.chunkedgraph.get_root_timestamps(oid)[0],
        exclude_links_to_future=True,
    )
    print(f"  Lineage graph has {len(lg['nodes'])} nodes and {len(lg['links'])} links")

# %%
