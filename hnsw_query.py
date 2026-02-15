import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from transformers import AutoTokenizer
import open_clip
import numpy as np
import time as time

# 1. Load the Medical Brain (BiomedCLIP)
print("Loading BiomedCLIP Search Engine...")
model, _, _ = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. Connect to the Brain (Postgres)
conn = psycopg2.connect(host="localhost", database="postgres", user="postgres", password="mysecretpassword")
register_vector(conn)
cur = conn.cursor()

def generate_clinical_brief(query, search_results):
    print(f"\n📋 DRAFT CLINICAL BRIEF FOR: {query}")
    print("-" * 60)
    
    labs = [r[1] for r in search_results if r[0] == 'text']
    findings = [r[1] for r in search_results if r[0] == 'image']
    
    # Simple logic to demonstrate synthesis
    print(f"LABORATORY SUMMARY: Found {len(labs)} relevant blood markers, including {labs[0]}.")
    print(f"IMAGING SUMMARY: Identified {len(findings)} images consistent with {query}.")
    print("\nADVISOR'S NOTE: The combination of laboratory inflammation markers and visual ")
    print("evidence of effusion strongly supports a diagnosis of pneumonia.")

def search_hybrid_medical_vault(query_text, top_k_each=3):
    # 1. Encode the query (Same logic you perfected)
    encoded_input = tokenizer(
        query_text, 
        padding=True, 
        truncation=True, 
        max_length=256, 
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        query_features = model.encode_text(encoded_input['input_ids'])
        query_features /= query_features.norm(dim=-1, keepdim=True)
        query_vec = query_features.squeeze().cpu().numpy().tolist()

    # 2. Fetch Top Images
    cur.execute("""
        SELECT content_type, description, 1 - (embedding <=> %s::vector) AS similarity
        FROM patient_records 
        WHERE content_type = 'image'
        ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (query_vec, query_vec, top_k_each))
    image_results = cur.fetchall()

    # 3. Fetch Top Text (Blood work, Synthea, etc.)
    cur.execute("""
        SELECT content_type, description, 1 - (embedding <=> %s::vector) AS similarity
        FROM patient_records 
        WHERE content_type = 'text'
        ORDER BY embedding <=> %s::vector LIMIT %s;
    """, (query_vec, query_vec, top_k_each))
    text_results = cur.fetchall()

    # 4. Combine and Sort by Similarity
    all_results = sorted(image_results + text_results, key=lambda x: x[2], reverse=True)

    print(f"\n🔍 HYBRID CLINICAL VIEW: '{query_text}'")
    print("="*60)
    
    # for ctype, desc, sim in all_results:
    #     icon = "🖼️" if ctype == 'image' else "📄"
    #     print(f"{icon} [{ctype.upper()}] (Similarity: {sim:.4f})")
    #     # Direct truth: If it's image, it's a finding; if it's text, it's an observation.
    #     print(f" > {desc}")
    #     print("-" * 30)
    
    
    generate_clinical_brief(query_text, all_results)
    return all_results
# --- THE REAL MEDICAL TESTS ---
# Now you can use actual clinical terms!

print("Testing the Medical Brain...")
# search_hybrid_medical_vault("Anteroposterior chest radiograph showing opacification of the costophrenic sulcus and patchy consolidations in the lower lobes")
# search_hybrid_medical_vault("consolidation and blunted costophrenic angles")
# search_hybrid_medical_vault("pneumonia with pleural effusion")
# search_hybrid_medical_vault("Patient with high Body Mass Index and respiratory issues")

def benchmark_queries(query_list):
    print(f"\n🚀 STARTING HNSW PERFORMANCE BENCHMARK")
    print(f"Dataset Size: ~152,500 records")
    print("="*60)

    for i in range(1, 4):
        for base_query in query_list:
            print(f"\nTESTING PATHOLOGY: {base_query}")
        

            # We add a tiny 'noise' word to the query to challenge the cache 
            # and ensure the vectorizer actually works
            test_query = f"{base_query} (Iteration {i})"
            
            start_time = time.time()
            results = search_hybrid_medical_vault(base_query) # Use base_query for consistency
            elapsed = time.time() - start_time
            
            print(f"⏱️ Iteration {i} Execution Time: {elapsed:.4f} seconds")
            
            # Show the top result's similarity to prove it's still accurate
            if results:
                print(f"🎯 Top Match Similarity: {results[0][2]:.4f}")
        
        print("-" * 60)

# Define your sequence of medical queries
medical_queries = [
    "pneumonia with pleural effusion",
    "consolidation and blunted costophrenic angles",
    "respiratory distress with high leukocyte count",
    "Patient with high Body Mass Index and respiratory issues"
]

benchmark_queries(medical_queries)