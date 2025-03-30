import os
import networkx as nx
import google.generativeai as genai
import json
from typing import List, Dict, Any

# Configure Gemini - ensure GOOGLE_API_KEY is set in your environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_entities_relationships(text: str) -> Dict[str, Any]:
    """Extract entities and relationships with robust error handling."""
    prompt = f"""Extract entities and relationships from this text, returning ONLY valid JSON:
    
    {{
        "entities": ["list", "of", "entities"],
        "relationships": [["subject", "relation", "object"]]
    }}

    Text to analyze: "{text}"
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        
        # Handle different response formats
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates'):
            response_text = response.candidates[0].content.parts[0].text
        else:
            raise ValueError("Unexpected response format")
        
        # Clean the response to get pure JSON
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        return json.loads(json_str)
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        return {"entities": [], "relationships": []}

def validate_relationships(text: str, relationships: List) -> List:
    """Validate relationships with better prompt engineering."""
    prompt = f"""Verify which relationships are correct in this text. 
    Return ONLY a JSON array of valid relationships:
    
    {{
        "valid_relationships": [["subject", "relation", "object"]]
    }}
    
    Text: "{text}"
    Relationships to validate: {relationships}
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        result = json.loads(response_text[json_start:json_end])
        
        return result.get("valid_relationships", [])
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return relationships  # Fallback to original

def build_knowledge_graph(text: str) -> nx.DiGraph:
    """Build knowledge graph with enhanced error handling."""
    try:
        data = extract_entities_relationships(text)
        valid_rels = validate_relationships(text, data.get("relationships", []))
        
        graph = nx.DiGraph()
        graph.add_nodes_from(data.get("entities", []))
        
        for rel in valid_rels:
            if len(rel) == 3:  # Ensure proper (subject, relation, object) format
                graph.add_edge(rel[0], rel[2], relation=rel[1])
        
        return graph
    except Exception as e:
        print(f"Graph construction error: {str(e)}")
        return nx.DiGraph()  # Return empty graph on failure

if __name__ == "__main__":
    sample_text = "Elon Musk founded SpaceX in 2002. Google acquired DeepMind in 2014."
    kg = build_knowledge_graph(sample_text)
    
    print("\nKnowledge Graph:")
    print("Nodes:", kg.nodes())
    print("Edges with relations:")
    for u, v, data in kg.edges(data=True):
        print(f"  {u} --{data['relation']}--> {v}")
    
    # Basic visualization (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(kg)
        nx.draw(kg, pos, with_labels=True, node_size=2000, node_color='skyblue')
        edge_labels = nx.get_edge_attributes(kg, 'relation')
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels)
        plt.show()
    except ImportError:
        print("\nInstall matplotlib for visualization: pip install matplotlib")
