import pandas as pd
from transformers import pipeline
import json
import os
import re

# --- CONFIGURATION ---
CSV_FILE_PATH = "/app/data/Sakri_Lahir dan Hastimurti_Gugur.csv"
OUTPUT_FILE_PATH = "/app/data/ner_results.json"

# Lowered threshold slightly to catch names like "Arya Basusara" (0.61)
CONFIDENCE_THRESHOLD = 0.60

# Words to ignore (Blocklist)
BLOCKLIST = {
    "Sang", "Para", "Yang", "Dan", "Di", "Ke", "Dari", "Saat", "Ketika", 
    "Maka", "Lalu", "Akan", "Telah", "Sudah", "Ia", "Dia", "Mereka",
    "Hal"
}

# --- PRE-PROCESSING FUNCTIONS ---
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?;:])(?=[a-zA-Z])', r'\1 ', text)
    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    return text.strip()

def clean_entity_name(word):
    word = word.replace("##", "")
    word = word.strip(" .,:;!?\"'()")
    word = re.sub(' +', ' ', word)
    return word

def expand_to_full_word(text, start, end):
    """
    Repairs cut-off names. 
    If the entity ends at index 'end', check if 'text[end]' is still a letter.
    If so, keep reading until we hit a space or punctuation.
    Example: "Arya Bas" -> "Arya Basusara"
    """
    # If we are already at the end of the text, return the current slice
    if end >= len(text):
        return text[start:end]

    # Check if we stopped in the middle of a word (next char is alphanumeric)
    current_end = end
    while current_end < len(text) and text[current_end].isalnum():
        current_end += 1
    
    return text[start:current_end]

def run_ner_process():
    print("--- STARTING NER PROCESS WITH AUTO-REPAIR ---")
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"ERROR: CSV file not found at {CSV_FILE_PATH}")
        return

    # 1. Load Data
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        df = df.dropna(subset=['Teks'])
        if 'Judul' not in df.columns:
            df['Judul'] = "Unknown Story"
        print(f"1. Loaded CSV. Found {len(df)} rows.")
    except Exception as e:
        print(f"   Error reading CSV: {e}")
        return

    # 2. Load Model
    print("2. Loading AI Model...")
    ner_pipeline = pipeline(
        "ner", 
        model="cahya/bert-base-indonesian-ner", 
        aggregation_strategy="simple" 
    )
    print("   Model loaded!")

    print(f"3. Processing rows...")
    unique_entities = {}
    total_rows = len(df)

    for index, row in df.iterrows():
        raw_text = str(row['Teks'])
        story_title = str(row['Judul']).strip()
        
        # Preprocess
        text = preprocess_text(raw_text)
        
        if len(text) < 5: continue
        if index % 10 == 0:
            print(f"   Processing row {index + 1}/{total_rows}...")

        try:
            results = ner_pipeline(text)
            
            for entity in results:
                group = entity['entity_group']
                score = float(entity['score'])
                start = entity['start']
                end = entity['end']

                # --- FILTER: Confidence ---
                if score < CONFIDENCE_THRESHOLD: continue

                # --- FIX: Auto-Repair Cut-off Names ---
                # Instead of trusting entity['word'], we extract from text with repair logic
                full_word_raw = expand_to_full_word(text, start, end)
                
                # Clean it up
                clean_word = clean_entity_name(full_word_raw)

                # --- FILTER: Junk ---
                if len(clean_word) < 3: continue
                # If the word is JUST a title (e.g. "Prabu"), ignore it. 
                # But "Prabu Basukesti" is fine because it contains a title inside a longer string.
                if clean_word in BLOCKLIST: continue 

                label = "Unknown"
                if group == 'PER': label = "Person"
                elif group == 'ORG': label = "Organization"
                elif group == 'LOC': label = "Location"
                
                if label != "Unknown":
                    if clean_word not in unique_entities:
                        unique_entities[clean_word] = {
                            "label": label, 
                            "score": score,
                            "stories": {story_title}
                        }
                    else:
                        unique_entities[clean_word]["stories"].add(story_title)
                        # Update score if better found
                        if score > unique_entities[clean_word]['score']:
                            unique_entities[clean_word]['score'] = score
                        
        except Exception as e:
            print(f"   Warning: Skipped row {index} due to error: {e}")

    # Format Output
    final_output = [
        {
            "name": name, 
            "label": data["label"], 
            "confidence": data["score"],
            "stories": sorted(list(data["stories"]))
        } 
        for name, data in unique_entities.items()
    ]
    final_output.sort(key=lambda x: x['name'])

    print(f"\n4. Saving results to: {OUTPUT_FILE_PATH}")
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"--- DONE! Found {len(final_output)} unique entities. ---")

if __name__ == "__main__":
    run_ner_process()