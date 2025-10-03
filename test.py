import json
import networkx as nx

# Load your KG JSON
with open("knowledge_graph.json", "r") as f:
    data = json.load(f)

# Build graph
G = nx.Graph()
G.add_nodes_from(data["entities"])

for rel in data.get("relationships", []):
    src, tgt, rel_type = rel["subject"], rel["object"], rel["relation"]
    G.add_edge(src, tgt, label=rel_type)


def query_graph(entity, G, max_results=20):
    if entity not in G:
        return f"No entity '{entity}' found in the knowledge graph."
    
    results = []
    for i, neighbor in enumerate(G.neighbors(entity)):
        if i >= max_results:
            break
        edge = G[entity][neighbor]
        results.append(f"{entity} --{edge['label']}--> {neighbor}")
    
    return "\n".join(results)
def fuzzy_query(term, G, max_results=10):
    matches = [n for n in G.nodes if term.lower() in n.lower()]
    if not matches:
        return f"No matches for '{term}'."
    
    entity = matches[0]  # take first match
    return query_graph(entity, G, max_results=max_results)

