"""Revision R8: are the consensus communities internally connected?

Traag, Waltman and van Eck (2019) show that Louvain can return communities that are badly connected or even
disconnected. This script counts, for every consensus community, the connected components of the subgraph
induced by its films, and the share of its films outside the largest of them.

Run from the project root:  python revisions/r08_community_connectivity.py
Writes outputs/tables/revisions/r08_community_connectivity.csv
"""
from pathlib import Path
import igraph as ig
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "processed" / "features" / "modeling_table_success_star_power.csv",
                 usecols=["node_id", "community_id"], low_memory=False)
labels = pd.read_csv(ROOT / "outputs" / "tables" / "community_manual_labeling_template_labeled.csv")
edges = pd.read_csv(ROOT / "processed" / "network" / "semantic_network_edges_mutual_knn.csv",
                    usecols=["source_node_id", "target_node_id"])
pos = pd.Series(np.arange(len(df)), index=df["node_id"])
g = ig.Graph(n=len(df), edges=np.c_[edges["source_node_id"].map(pos), edges["target_node_id"].map(pos)].astype(int).tolist())
rows = []
for c, members in df.groupby("community_id").groups.items():
    sub = g.induced_subgraph(pos[df.loc[members, "node_id"]].values.tolist())
    comps = sub.connected_components()
    rows.append({"community_id": c, "films": sub.vcount(), "components": len(comps),
                 "films_outside_largest_component": sub.vcount() - max(comps.sizes()),
                 "major": c in set(labels["community_id"])})
t = pd.DataFrame(rows)
t.to_csv(ROOT / "outputs" / "tables" / "revisions" / "r08_community_connectivity.csv", index=False)
for name, s in [("all 105", t), ("35 major", t[t["major"]]), ("70 minor", t[~t["major"]])]:
    print(f"{name}: connected {int((s['components'] == 1).sum())}/{len(s)}; films outside largest component "
          f"{int(s['films_outside_largest_component'].sum())} ({s['films_outside_largest_component'].sum() / s['films'].sum():.2%})")
print(t[t["components"] > 1].sort_values("films_outside_largest_component", ascending=False).head(10).to_string(index=False))
