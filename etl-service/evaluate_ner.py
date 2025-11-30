import json
import os

# --- CONFIGURATION ---
RESULTS_FILE_PATH = "/app/data/ner_results_NusaBert-ner-v1.3.json"

# --- GOLD STANDARD (Ground Truth) ---
# This list represents the "Correct Answers".
GOLD_STANDARD = [
    # PERSON
    {"name": "Prabu Basukesti", "label": "Person"},
    {"name": "Patih Jayaloka", "label": "Person"},
    {"name": "Resi Suganda", "label": "Person"},
    {"name": "Empu Dewayasa", "label": "Person"},
    {"name": "Empu Purbageni", "label": "Person"},
    {"name": "Empu Prawa", "label": "Person"},
    {"name": "Empu Kanomayasa", "label": "Person"},
    {"name": "Dewi Kaniraras", "label": "Person"},
    {"name": "Dewi Marapi", "label": "Person"},
    {"name": "Resi Kuswala", "label": "Person"},
    {"name": "Bambang Daneswara", "label": "Person"},
    {"name": "Prabu Cingkaradewa", "label": "Person"},
    {"name": "Sri Maharaja Purwacandra", "label": "Person"},
    {"name": "Ditya Citradana", "label": "Person"},
    {"name": "Putut Margana", "label": "Person"},
    {"name": "Indramarkata", "label": "Person"},
    {"name": "Kalayaksa", "label": "Person"},
    {"name": "Gajah Barigu", "label": "Person"},
    {"name": "Garuda Urna", "label": "Person"},
    {"name": "Naga Wiswana", "label": "Person"},
    {"name": "Resi Manumanasa", "label": "Person"},
    {"name": "Bambang Satrukem", "label": "Person"},
    {"name": "Dewi Nilawati", "label": "Person"},
    {"name": "Janggan Smara", "label": "Person"},
    {"name": "Prabu Durapati", "label": "Person"},
    {"name": "Prabu Basupati", "label": "Person"},
    {"name": "Prabu Hastimurti", "label": "Person"},
    {"name": "Resi Basunanda", "label": "Person"},
    {"name": "Patih Basundara", "label": "Person"},
    {"name": "Raden Wasanta", "label": "Person"},
    {"name": "Arya Basusara", "label": "Person"},
    {"name": "Prabu Daneswara", "label": "Person"},
    {"name": "Dewi Awanti", "label": "Person"},
    {"name": "Brahmana Wisaka", "label": "Person"},
    {"name": "Prabu Sriwahana", "label": "Person"},

    # LOCATION
    {"name": "Utarakanda", "label": "Location"},
    {"name": "Purwakanda", "label": "Location"},
    {"name": "Daksinakanda", "label": "Location"},
    {"name": "Pracimakanda", "label": "Location"},
    {"name": "Padepokan Saptaarga", "label": "Location"},
    {"name": "Gunung Saptaarga", "label": "Location"},
    {"name": "Tanah Hindustan", "label": "Location"},
    {"name": "Tanah Jawa", "label": "Location"},
    {"name": "Hutan Minangsraya", "label": "Location"},
    {"name": "Medang Kamulan", "label": "Location"},

    # ORGANIZATION
    {"name": "Kerajaan Wirata", "label": "Organization"},
    {"name": "Kerajaan Duhyapura", "label": "Organization"},
    {"name": "Kerajaan Gajahoya", "label": "Organization"},
    {"name": "Kerajaan Medang Kamulan", "label": "Organization"},
    {"name": "Gilingwesi", "label": "Organization"}
]

def calculate_metrics():
    print("--- COMPREHENSIVE EVALUATION REPORT ---")
    
    if not os.path.exists(RESULTS_FILE_PATH):
        print(f"Error: {RESULTS_FILE_PATH} not found.")
        return

    # 1. Load Results
    with open(RESULTS_FILE_PATH, 'r') as f:
        json_results = json.load(f)
    
    # Create dictionaries for fast lookup (lowercase key -> object)
    # We map "name.lower()" to the full object so we can check labels
    model_predictions = {item['name'].lower(): item for item in json_results}
    gold_truth = {item['name'].lower(): item for item in GOLD_STANDARD}

    # 2. Calculate Counts
    tp = 0  # True Positive: Found & Correct Label
    fp = 0  # False Positive: Found but Wrong Label (or not in Gold)
    fn = 0  # False Negative: In Gold but Not Found
    
    # For "Strict" metrics (Fairer for partial gold standard)
    strict_tp = 0
    strict_fp_label_error = 0
    
    # --- Analyze Gold Standard Items (Recall focus) ---
    print(f"\n1. Analyzing Coverage of {len(GOLD_STANDARD)} Gold Items...")
    for name_lower, gold_item in gold_truth.items():
        if name_lower in model_predictions:
            prediction = model_predictions[name_lower]
            
            # Check Label match
            if prediction['label'].lower() == gold_item['label'].lower():
                tp += 1
                strict_tp += 1
            else:
                # It was found, but label is wrong (e.g. "Medang Kamulan" labeled Person instead of Location)
                # This counts as a False Positive for the label, and False Negative for the entity
                print(f"   [MISMATCH] '{gold_item['name']}' | Gold: {gold_item['label']} vs Model: {prediction['label']}")
                strict_fp_label_error += 1
                fn += 1 
        else:
            # Not found at all
            # print(f"   [MISSED] '{gold_item['name']}'")
            fn += 1

    # --- Analyze Model Predictions (Precision focus) ---
    # NOTE: Since our Gold Standard is partial, we can only judge False Positives 
    # relative to the items we KNOW about.
    
    # 3. Calculate Metrics
    
    # --- A. RECALL (Completeness) ---
    # Formula: TP / (TP + FN)
    # "Did we find everything in the Gold List?"
    recall = (tp / len(GOLD_STANDARD)) * 100 if len(GOLD_STANDARD) > 0 else 0

    # --- B. STRICT PRECISION (Correctness) ---
    # Formula: TP / (TP + Label Errors)
    # "Of the Gold items we found, how often was the label correct?"
    # This ignores "extra" items the model found that aren't in our list.
    total_found_in_gold_set = strict_tp + strict_fp_label_error
    strict_precision = (strict_tp / total_found_in_gold_set) * 100 if total_found_in_gold_set > 0 else 0
    
    # --- C. STRICT F1 SCORE ---
    # Harmonic mean of Recall and Strict Precision
    if (strict_precision + recall) > 0:
        f1 = 2 * (strict_precision * recall) / (strict_precision + recall)
    else:
        f1 = 0

    # --- D. LABEL ACCURACY ---
    # "Accuracy" in NER is tricky. Usually, we look at Label Accuracy.
    # This is: Correct Labels / Total Matched Entities
    label_accuracy = strict_precision # In this context, it's the same.

    # 4. Output Report
    print("\n" + "="*40)
    print("       METRICS REPORT")
    print("="*40)
    print(f"Gold Standard Size : {len(GOLD_STANDARD)}")
    print(f"Matches Found      : {total_found_in_gold_set}")
    print(f"Perfect Matches (TP): {strict_tp}")
    print(f"Label Errors       : {strict_fp_label_error}")
    print(f"Completely Missed  : {fn}")
    print("-" * 40)
    print(f"RECALL             : {recall:.2f}%")
    print("   (How many of the Gold items did we find?)")
    print("-" * 40)
    print(f"PRECISION (Strict) : {strict_precision:.2f}%")
    print("   (When we found a Gold item, was the label correct?)")
    print("-" * 40)
    print(f"F1 SCORE           : {f1:.2f}%")
    print("   (The balance between Recall and Precision)")
    print("-" * 40)
    print(f"LABEL ACCURACY     : {label_accuracy:.2f}%")
    print("="*40)

    # 5. Diagnostics
    if label_mismatch_list := [k for k,v in gold_truth.items() if k in model_predictions and model_predictions[k]['label'].lower() != v['label'].lower()]:
        print("\n[!] LABEL ERRORS DETECTED:")
        for name in label_mismatch_list:
            print(f"    - {gold_truth[name]['name']}: Model thought it was {model_predictions[name]['label']}")

if __name__ == "__main__":
    calculate_metrics()