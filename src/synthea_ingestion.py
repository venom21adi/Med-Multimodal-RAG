import pandas as pd
import psycopg2
from pgvector.psycopg2 import register_vector
import torch
import hashlib
import json
import open_clip
from transformers import AutoTokenizer

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


# 1. Setup BiomedCLIP
print("Loading BiomedCLIP...")
model, _, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
# BYPASS the open_clip tokenizer and use the HF version directly
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. Connect to DB
register_vector(conn)
cur = conn.cursor()

# 3. Load Synthea Data
csv_path = "source_file_path"
df = pd.read_csv(csv_path)
df_sample = df.head(1000)

print(f"Ingesting {len(df_sample)} medical text records using BiomedCLIP...")

for index, row in df_sample.iterrows():
    # Keep your interpretability, but we'll use a cleaner description for the embedding
    description = f"Observation: {row['DESCRIPTION']} value: {row['VALUE']} {row['UNITS']} on date {row['DATE']}"
    patient_id = row['PATIENT']
    unique_row_id = get_hash(f"{patient_id}_{description}")

    # BiomedCLIP uses a 256-token context window (much better than standard CLIP's 77)
    encoded_input = tokenizer(
            description, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors='pt'
        ).to(device)

    with torch.no_grad():
        # Generate the text embedding
        text_features = model.encode_text(encoded_input['input_ids'])
        
        # CRITICAL: Normalize for Cosine Similarity so it matches image scaling
        text_features /= text_features.norm(dim=-1, keepdim=True)
        embedding = text_features.squeeze().cpu().numpy().tolist()

    if index == 0:
        print(f"DEBUG: Confirmed Medical Embedding size: {len(embedding)}")

    metadata = {
        "code": row['CODE'],
        "date": row['DATE'],
        "original_id": patient_id,
        "model": "BiomedCLIP-PubMedBERT"
    }

    try:
        cur.execute("""
            INSERT INTO patient_records (patient_id, content_type, metadata, description, embedding, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING;
        """, (patient_id, 'text', json.dumps(metadata), description, embedding, unique_row_id))
        
        if index % 100 == 0:
            conn.commit()
            print(f"Text Progress: {index}/{len(df_sample)}...")
    except Exception as e:
        print(f"Error at index {index}: {e}")
        conn.rollback()

conn.commit()
print("Success! Text and Images are now unified in the Medical Latent Space.")
