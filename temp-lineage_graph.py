import matplotlib.pyplot as plt
import networkx as nx
from caveclient import CAVEclient

# -------------------------------------------------
# Configuration
# -------------------------------------------------
DATASET = "stroeh_mouse_retina"
SEG_ID = 720575940564522340

# -------------------------------------------------
# Fetch lineage graph
# -------------------------------------------------
client = CAVEclient(DATASET)
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

# Print for visibility (same behavior as notebook cell output)
print(original_ids)
