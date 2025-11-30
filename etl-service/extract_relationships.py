import pandas as pd
import google.generativeai as genai
import json
import os
import time

# --- CONFIGURATION ---
CSV_FILE_PATH = "/app/data/Sakri_Lahir dan Hastimurti_Gugur.csv"
NER_FILE_PATH = "/app/data/ner_results_NusaBert-ner-v1.3.json"
OUTPUT_FILE_PATH = "/app/data/relationships_NusaBert-ner-v1.3.json"

# Load API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') # Flash is faster/cheaper for this

def load_valid_entities():
    """Loads the entities we found in the previous step to use as a constraint."""
    if not os.path.exists(NER_FILE_PATH):
        print("Error: ner_results.json not found. Run NER step first!")
        return []
    
    with open(NER_FILE_PATH, 'r') as f:
        data = json.load(f)
    
    # Return just the list of names to keep the prompt small
    # We filter out very low confidence junk if needed
    return [item['name'] for item in data if item['confidence'] > 0.6]

def get_relationships_from_gemini(text, valid_entities):
    """Sends text to Gemini to find edges."""
    
    # We split valid entities into a string for the prompt
    entities_str = ", ".join(valid_entities)

    prompt = f"""
    You are a Graph Database expert for Wayang stories.
    
    **Task:** Extract relationships from the STORY TEXT below.
    
    **Constraint:** You must ONLY use entity names from this VALID ENTITY LIST:
    [{entities_str}]
    
    If a name in the text (like "Basukesti") matches a name in the list (like "Prabu Basukesti"), USE THE LIST NAME.
    Ignore entities that are not in the list.

    **Allowed Relationships:**
    - FATHER_OF, MOTHER_OF, SON_OF, DAUGHTER_OF
    - SIBLING_OF, MARRIED_TO
    - KILLED_BY, ALLY_OF, ENEMY_OF
    - KING_OF, LEADER_OF
    - LOCATED_IN (for Person -> Location)
    - USES_WEAPON (for Person -> Weapon)
    - MENTOR_OF, STUDENT_OF

    **Format:**
    Return ONLY a JSON list of objects. Do not write markdown code blocks.
    Example:
    [
        {{"source": "Prabu Basukesti", "target": "Kerajaan Wirata", "type": "KING_OF"}},
        {{"source": "Bambang Sakri", "target": "Bambang Satrukem", "type": "SON_OF"}}
    ]

    **STORY TEXT:**
    {text}
    """

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        
        # Clean up potential markdown formatting from LLM
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "")
        
        return json.loads(raw_text)
    except Exception as e:
        print(f"   [!] Error getting relationships: {e}")
        return []

def run_extraction():
    print("--- STARTING RELATIONSHIP EXTRACTION ---")
    
    # 1. Load Context (Nodes)
    print("1. Loading Valid Entities...")
    valid_entities = load_valid_entities()
    print(f"   Loaded {len(valid_entities)} valid entities to guide Gemini.")

    # 2. Load Data (Text)
    print("2. Loading Story Text...")
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df = df.dropna(subset=['Teks'])
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        return

    all_relationships = []

    # 3. Process Rows
    print("3. asking Gemini to find connections (Row by Row)...")
    total = len(df)
    
    for index, row in df.iterrows():
        text = str(row['Teks'])
        story_title = str(row.get('Judul', 'Unknown'))
        
        if len(text) < 50: continue
        
        print(f"   Processing Row {index+1}/{total} ({story_title})...")
        
        # Call Gemini
        rels = get_relationships_from_gemini(text, valid_entities)
        
        # Add metadata
        for r in rels:
            r['story_source'] = story_title
            all_relationships.append(r)
            
        # Sleep briefly to respect rate limits
        time.sleep(1)

    # 4. Save Results
    print(f"\n4. Saving {len(all_relationships)} relationships to {OUTPUT_FILE_PATH}")
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(all_relationships, f, indent=2)
    
    print("--- EXTRACTION COMPLETE ---")

if __name__ == "__main__":
    run_extraction()