import os
import fitz  # PyMuPDF for PDF extraction
import networkx as nx
import google.generativeai as genai
import json
from typing import List, Dict, Any

# Configure Gemini - Ensure GOOGLE_API_KEY is set in your environment
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    """Break large text into smaller chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def extract_entities_relationships(text: str) -> Dict[str, Any]:
    """Extract entities and relationships from text using Gemini API."""
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
        
        # Handle response formats
        response_text = response.text if hasattr(response, 'text') else json.dumps({"entities": [], "relationships": []})
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        json_str = response_text[json_start:json_end]
        
        return json.loads(json_str)
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        return {"entities": [], "relationships": []}

def build_knowledge_graph(pdf_path: str) -> nx.DiGraph:
    """Build knowledge graph from financial report PDF."""
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        
        graph = nx.DiGraph()
        all_entities = set()
        all_relationships = []

        for chunk in chunks:
            data = extract_entities_relationships(chunk)
            
            # Add nodes (entities)
            entities = data.get("entities", [])
            graph.add_nodes_from(entities)
            all_entities.update(entities)

            # Add edges (relationships)
            for rel in data.get("relationships", []):
                if len(rel) == 3:
                    graph.add_edge(rel[0], rel[2], relation=rel[1])
                    all_relationships.append({"subject": rel[0], "relation": rel[1], "object": rel[2]})

        return graph, list(all_entities), all_relationships
    except Exception as e:
        print(f"Graph construction error: {str(e)}")
        return nx.DiGraph(), [], []

if __name__ == "__main__":
    pdf_path = "financial_report.pdf"  # Update this with the actual file path
    kg, entities, relationships = build_knowledge_graph(pdf_path)

    # Print summary
    print("\nKnowledge Graph Stats:")
    print(f"Total Entities: {len(entities)}")
    print(f"Total Relationships: {len(relationships)}\n")

    print("Entities:")
    print(entities)

    print("\nRelationships:")
    for rel in relationships:
        print(f"{rel['subject']} --{rel['relation']}--> {rel['object']}")

    # Save to JSON file
    output_data = {
        "total_entities": len(entities),
        "total_relationships": len(relationships),
        "entities": entities,
        "relationships": relationships
    }

    with open("knowledge_graph.json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)

    print("\n✅ Knowledge graph data saved to 'knowledge_graph.json'")

    # Visualization
    try:
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(kg)
        nx.draw(kg, pos, with_labels=True, node_size=2000, node_color='skyblue')
        edge_labels = nx.get_edge_attributes(kg, 'relation')
        nx.draw_networkx_edge_labels(kg, pos, edge_labels=edge_labels)
        plt.show()
    except ImportError:
        print("\n⚠️ Install matplotlib for visualization: pip install matplotlib")

