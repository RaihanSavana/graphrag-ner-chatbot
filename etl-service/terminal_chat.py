import google.generativeai as genai
from neo4j import GraphDatabase
import os
import json
import csv
import numpy as np  # Required for vector math

# --- CONFIG ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
CSV_FILENAME = "/app/data/Sakri_Lahir dan Hastimurti_Gugur.csv"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
# Use Flash for text generation, but we will use 'text-embedding-004' for vectors
model = genai.GenerativeModel('gemini-2.5-flash') 

# --- 1. SEMANTIC VECTOR SEARCH COMPONENT (The Upgrade) ---
class SemanticVectorDB:
    def __init__(self, filename):
        self.documents = [] # Stores the actual text
        self.embeddings = None # Stores the vector numbers
        self.filename = filename
        self.load_and_embed()

    def load_and_embed(self):
        if not os.path.exists(self.filename):
            print(f"⚠️ Warning: {self.filename} not found.")
            return
            
        # 1. Read CSV
        raw_texts = []
        with open(self.filename, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create a rich context string
                text_blob = f"Title: {row.get('Judul')} | Sub: {row.get('Subjudul')}\nContent: {row.get('Teks')}"
                self.documents.append(text_blob)
                raw_texts.append(text_blob)
        
        if not self.documents: return

        # 2. Generate Embeddings using Gemini
        print(f"   [VectorDB] Generating semantic embeddings for {len(self.documents)} docs...")
        try:
            # We pass the list of texts to embed them in batch
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=raw_texts,
                task_type="retrieval_document",
                title="Wayang Knowledge Base"
            )
            # Convert to numpy array for fast math
            self.embeddings = np.array(result['embedding'])
            print("   [VectorDB] Embeddings ready & stored in memory.")
        except Exception as e:
            print(f"   [VectorDB] Error generating embeddings: {e}")

    def search(self, query, top_k=3):
        """
        Calculates Cosine Similarity between Query Embedding and Document Embeddings
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []

        try:
            # 1. Embed the Query
            query_res = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            q_vector = np.array(query_res['embedding'])
            
            # 2. Calculate Dot Product (Cosine Similarity)
            # Since Gemini embeddings are normalized, dot product == cosine similarity
            scores = np.dot(self.embeddings, q_vector)
            
            # 3. Get Top-K Indices
            # argsort sorts ascending, so we take the last k and reverse them
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = scores[idx]
                doc = self.documents[idx]
                # debug print to show semantic score
                # print(f"      > Score: {score:.4f} | Doc: {doc[:50]}...") 
                results.append(doc)
                
            return results
        except Exception as e:
            print(f"Search Error: {e}")
            return []

# Initialize the Semantic DB
vector_db = SemanticVectorDB(CSV_FILENAME)

# --- 2. SCHEMA DEFINITIONS & GRAPH SEARCH ---
VALID_RELATIONSHIPS = [
    "KING_OF", "ALLY_OF", "FATHER_OF", "SON_OF", "MARRIED_TO", 
    "DAUGHTER_OF", "MOTHER_OF", "SIBLING_OF", "SIBILING_OF", 
    "MENTOR_OF", "LEADER_OF", "ENEMY_OF", "KILLED_BY", 
    "USES_WEAPON", "LOCATED_IN"
]

def get_graph_context(question):
    schema_str = f"""
    Node Properties: name (String)
    Relationship Types: {', '.join(VALID_RELATIONSHIPS)}
    """

    cypher_prompt = f"""
    Task: Generate a Neo4j Cypher query.
    STRICT SCHEMA: {schema_str}
    RULES:
    1. Use 'WHERE toLower(n.name) CONTAINS toLower("...")' for names.
    2. Follow direction: (Victim)-[:KILLED_BY]->(Killer), (Child)-[:SON_OF]->(Parent).
    3. Return keys: source, relationship, target.
    Question: "{question}"
    Output: Cypher query ONLY.
    """
    
    cypher_query = ""
    try:
        res = model.generate_content(cypher_prompt)
        cypher_query = res.text.strip().replace("```cypher", "").replace("```", "").strip()
    except Exception as e:
        return [], f"Gen Error: {e}"

    results = []
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run(cypher_query)
            results = [dict(r) for r in result]
    except Exception as e:
        results = [f"Neo4j Error: {str(e)}"]
    finally:
        if driver: driver.close()
    
    return results, cypher_query

# --- 3. PROMPT BUILDER & ORCHESTRATOR ---
def build_hybrid_prompt(question, vector_data, graph_data):
    return f"""
    User Question: {question}
    
    [SOURCE 1: KNOWLEDGE GRAPH (Structured Facts)]
    {json.dumps(graph_data, indent=2)}
    
    [SOURCE 2: SEMANTIC DOCUMENTS (Contextual Story)]
    {json.dumps(vector_data, indent=2)}
    
    Instruction: Synthesize an answer in Indonesian.
    - Use the Graph for precise relationships (Who is father of whom).
    - Use the Documents for narrative details (How did it happen).
    """

def generate_answer(context):
    return model.generate_content(context).text.strip()

# --- MAIN LOOP ---
if __name__ == "__main__":
    print("\n🤖 HYBRID WAYANG BOT (Semantic + Graph)\n")
    while True:
        q = input("User: ")
        if q.lower() in ['exit', 'quit']: break
        
        # Parallel Retrieval
        print("   ... Searching Graph & Vectors ...")
        v_res = vector_db.search(q)  # Now uses Gemini Embeddings
        g_res, cypher = get_graph_context(q)
        
        # Synthesis
        final_prompt = build_hybrid_prompt(q, v_res, g_res)
        print("   ... Synthesizing ...")
        ans = generate_answer(final_prompt)
        
        print(f"Bot: {ans}\n")
        # Optional Debug
        # print(f"   [Debug Cypher]: {cypher}")