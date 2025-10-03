import os
import fitz  # PyMuPDF
import networkx as nx
import google.generativeai as genai
import time
import json
import hashlib
import pickle
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
PDF_PATH = "tesla.pdf"
CACHE_DIR = "cache_chunks"
CHUNK_SIZE = 5000  # bigger chunks → fewer API calls
MAX_WORKERS = 5    # parallel requests

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === Ensure cache directory exists ===
os.makedirs(CACHE_DIR, exist_ok=True)

# === PDF extraction ===
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# === Chunking ===
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# === Caching ===
def cache_result(chunk: str, data: dict):
    key = hashlib.md5(chunk.encode()).hexdigest()
    with open(os.path.join(CACHE_DIR, f"{key}.pkl"), "wb") as f:
        pickle.dump(data, f)

def load_cache(chunk: str):
    key = hashlib.md5(chunk.encode()).hexdigest()
    try:
        with open(os.path.join(CACHE_DIR, f"{key}.pkl"), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# === Extract entities and relationships with Gemini ===
def extract_entities_relationships(chunk: str, max_retries: int = 3) -> Dict[str, Any]:
    cached = load_cache(chunk)
    if cached:
        return cached

    prompt = f"""
Extract entities and relationships from this text and return ONLY valid JSON in the format:
{{
    "entities": ["list", "of", "entities"],
    "relationships": [["subject", "relation", "object"]]
}}
Text to analyze:
\"\"\"{chunk}\"\"\"
"""

    attempt = 0
    while attempt < max_retries:
        try:
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(prompt)
            response_text = getattr(response, "text", "{}")

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                data = {"entities": [], "relationships": []}

            cache_result(chunk, data)
            return data

        except Exception as e:
            attempt += 1
            wait = 2 ** attempt
            print(f"⚠️ Extraction error: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    print("⚠️ Max retries reached. Returning empty results.")
    return {"entities": [], "relationships": []}

# === Build knowledge graph in parallel ===
def build_knowledge_graph(pdf_path: str) -> nx.DiGraph:
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    graph = nx.DiGraph()
    all_entities = set()
    all_relationships = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_entities_relationships, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            data = future.result()
            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            if entities:
                graph.add_nodes_from(entities)
                all_entities.update(entities)

            for rel in relationships:
                if len(rel) == 3:
                    graph.add_edge(rel[0], rel[2], relation=rel[1])
                    all_relationships.append({"subject": rel[0], "relation": rel[1], "object": rel[2]})

    return graph, list(all_entities), all_relationships

# === Simple RAG query interface over graph ===
def query_graph(graph: nx.DiGraph, query_entity: str) -> List[str]:
    """
    Returns textual answers for a query entity by traversing relationships.
    """
    answers = []
    if query_entity in graph.nodes:
        for neighbor in graph.neighbors(query_entity):
            rel = graph[query_entity][neighbor].get("relation", "")
            answers.append(f"{query_entity} --{rel}--> {neighbor}")

        for pred in graph.predecessors(query_entity):
            rel = graph[pred][query_entity].get("relation", "")
            answers.append(f"{pred} --{rel}--> {query_entity}")
    else:
        answers.append(f"No information found for '{query_entity}' in the graph.")
    return answers

# === Main execution ===
if __name__ == "__main__":
    kg, entities, relationships = build_knowledge_graph(PDF_PATH)

    print(f"\nTotal Entities: {len(entities)}")
    print(f"Total Relationships: {len(relationships)}\n")

    # Save to JSON
    with open("knowledge_graph.json", "w") as f:
        json.dump({
            "total_entities": len(entities),
            "total_relationships": len(relationships),
            "entities": entities,
            "relationships": relationships
        }, f, indent=4)

    print("✅ Knowledge graph saved to 'knowledge_graph.json'")

    # Example query
    query = "Tesla"
    answers = query_graph(kg, query)
    print(f"\nRAG Query Results for '{query}':")
    for ans in answers:
        print(ans)

    # Optional visualization
    try:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(kg, seed=42)
        nx.draw(kg, pos, with_labels=True, node_size=2000, node_color='skyblue')
        edge_labels = nx.get_edge_attributes(kg, 'relation')
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels)
        plt.show()
    except ImportError:
        print("⚠️ Install matplotlib for visualization: pip install matplotlib")
