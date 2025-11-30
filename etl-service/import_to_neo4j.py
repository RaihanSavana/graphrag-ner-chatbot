import json
import os
import time
from neo4j import GraphDatabase
# --- CONFIGURATION ---
NER_FILE_PATH = "/app/data/ner_results.json"
REL_FILE_PATH = "/app/data/relationships_NusaBert-ner-v1.3.json"
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
def wait_for_neo4j(driver):
    print("Connecting to Neo4j...")
    for i in range(30):
        try:
            driver.verify_connectivity()
            print("   Neo4j is ready!")
            return True
        except Exception as e:
            print(f"   Waiting for Neo4j... ({e})")
            time.sleep(1)
    return False
def run_import():
    print("--- STARTING NEO4J IMPORT ---")
    # 1. Load JSON Data
    if not os.path.exists(NER_FILE_PATH) or not os.path.exists(REL_FILE_PATH):
        print("Error: JSON data files not found. Run NER and ER scripts first.")
        return
    with open(NER_FILE_PATH, 'r') as f:
        nodes = json.load(f)
    
    with open(REL_FILE_PATH, 'r') as f:
        rels = json.load(f)
    # 2. Connect to DB
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    if not wait_for_neo4j(driver): return
    with driver.session() as session:
        
        # 3. Import Nodes
        print(f"1. Importing {len(nodes)} Nodes...")
        for node in nodes:
            # We handle multiple labels if needed, but simplified here
            cypher = f"MERGE (n:{node['label']} {{name: $name}})"
            session.run(cypher, name=node['name'])
            
            # Optional: Add a 'Story' label or property
            for story in node.get('stories', []):
                # Add label for the story? or just property? Let's add a property 'stories'
                session.run(
                    f"MATCH (n:{node['label']} {{name: $name}}) "
                    "SET n.stories = $stories",
                    name=node['name'], stories=node.get('stories', [])
                )
        
        # 4. Import Relationships
        print(f"2. Importing {len(rels)} Relationships...")
        for rel in rels:
            source = rel.get('source')
            target = rel.get('target')
            rel_type = rel.get('type', 'RELATED_TO').upper().replace(" ", "_")
            
            if not source or not target: continue
            # Cypher to connect them
            # We strictly match on Name
            query = f"""
            MATCH (a {{name: $source}})
            MATCH (b {{name: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.source_story = $story
            """
            session.run(query, source=source, target=target, story=rel.get('story_source', 'Unknown'))
    driver.close()
    print("--- IMPORT COMPLETE! Check http://localhost:7474 ---")
if __name__ == "__main__":
    run_import()