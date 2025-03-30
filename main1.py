import os
import networkx as nx
import google.generativeai as genai
import json
from typing import Dict, List, Any
import re  # For better JSON extraction

# Configure Gemini - set GOOGLE_API_KEY in your environment first
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_json_from_response(response_text: str) -> Dict:
    """Robust JSON extraction from LLM response"""
    try:
        # First try direct parse
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback: extract first JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError("No valid JSON found in response")

def extract_and_validate(text: str) -> Dict[str, Any]:
    """Single LLM call that extracts and validates entities/relationships"""
    prompt = f"""Perform these tasks on the text below:
    1. Extract all important entities
    2. Identify relationships between entities
    3. Validate that relationships are factually correct
    
    Return ONLY this JSON format:
    {{
        "entities": ["entity1", "entity2"],
        "valid_relationships": [["subject", "relation", "object"]]
    }}
    
    Text: "{text}"
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        
        # Handle different response formats
        response_text = response.text if hasattr(response, 'text') else \
                      response.candidates[0].content.parts[0].text
        
        data = extract_json_from_response(response_text)
        
        # Validate structure
        if not all(key in data for key in ["entities", "valid_relationships"]):
            raise ValueError("Missing required fields in response")
            
        return {
            "entities": list(set(data["entities"])),  # Remove duplicates
            "valid_relationships": [
                rel for rel in data["valid_relationships"] 
                if len(rel) == 3  # Ensure proper triplet format
            ]
        }
        
    except Exception as e:
        print(f"Processing error: {str(e)}")
        return {"entities": [], "valid_relationships": []}

def build_knowledge_graph(text: str) -> nx.DiGraph:
    """Build knowledge graph with just 1 LLM call"""
    data = extract_and_validate(text)
    
    graph = nx.DiGraph()
    graph.add_nodes_from(data["entities"])
    
    for subj, rel, obj in data["valid_relationships"]:
        graph.add_edge(subj, obj, relation=rel)
    
    return graph

def visualize_graph(graph: nx.DiGraph):
    """Optional visualization with matplotlib"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph, seed=42)
        
        nx.draw(graph, pos, with_labels=True, 
               node_size=2000, node_color='skyblue',
               font_size=10, font_weight='bold')
        
        edge_labels = {(u, v): d['relation'] 
                      for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, 
                                   edge_labels=edge_labels,
                                   font_color='red')
        
        plt.title("Knowledge Graph", size=15)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Visualization skipped: pip install matplotlib")

if __name__ == "__main__":
    sample_text = """Elon Musk founded SpaceX in 2002. 
    Google acquired DeepMind in 2014. 
    Microsoft invested in OpenAI."""
    
    print("Processing text...")
    kg = build_knowledge_graph(sample_text)
    
    print("\n=== Knowledge Graph ===")
    print(f"Nodes ({len(kg.nodes())}):", kg.nodes())
    print(f"Relationships ({len(kg.edges())}):")
    for u, v, data in kg.edges(data=True):
        print(f"  {u} --{data['relation']}--> {v}")
    
    visualize_graph(kg)
