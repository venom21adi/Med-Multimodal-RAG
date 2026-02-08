import os
import torch
import psycopg2
from pgvector.psycopg2 import register_vector
from PIL import Image
import json
import hashlib
import open_clip
import pandas as pd

# 1. Setup BiomedCLIP (The Medical Brain)
print("Loading BiomedCLIP...")
# This model uses PubMedBERT for text and ViT-B-16 for images
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)
tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. DB Connection

register_vector(conn)
cur = conn.cursor()



# 3. Path Configuration
image_folder = r"image_path"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))][:500]

print(f"Ingesting {len(image_files)} images using BiomedCLIP medical embeddings...")

label_df = pd.read_csv(r"C:\RAG\Source\archive\sample\sample_labels.csv")

for img_name in image_files:
    clinical_findings = label_df.loc[label_df['Image Index'] == img_name, 'Finding Labels'].values[0]
    clean_labels = clinical_findings.replace('|', ', ')
    text_to_embed = f"A chest X-ray showing {clean_labels}"
    # BiomedCLIP/PubMedBERT has a 256 token limit
    # We manually handle the input_ids conversion
    encoded_text = tokenizer.tokenizer(
        text_to_embed, 
        padding='max_length', 
        truncation=True, 
        max_length=256, 
        return_tensors='pt'
    ).to(device)
    try:
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Unique hash for deduplication
        with open(img_path, "rb") as f:
            img_hash = hashlib.md5(f.read()).hexdigest()

        # Generate BiomedCLIP Image Embedding
        # Preprocess converts the image to the specific 224x224 format the model expects
        image_input = preprocess_val(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            # IMPORTANT: Normalize the vector. Medical RAG requires precise unit vectors.
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # 3. Get Textual Features (The "Ground Truth")
            # 4. Get Textual Features (Using input_ids directly)
            text_features = model.encode_text(encoded_text['input_ids'])
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 4. FUSE: 70% Image, 30% Label (Adjust weighting as needed)
            # This ensures the image stays primary but is "pulled" toward the correct diagnosis
            fused_features = (0.7 * image_features) + (0.3 * text_features)
            fused_features /= fused_features.norm(dim=-1, keepdim=True)

            embedding = fused_features.squeeze().cpu().numpy().tolist()
       
       
        rich_description = f"Chest X-ray findings: {clinical_findings.replace('|', ', ')}. Filename: {img_name}"
        # Metadata
        metadata = json.dumps({
            "source": "NIH_Dataset_Medical_CLIP",
            "filename": img_name,
            "model": "BiomedCLIP-ViT-B-16"
        })

        cur.execute("""
            INSERT INTO patient_records (patient_id, content_type, metadata, description, embedding, content_hash)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO NOTHING;
        """, (img_name, 'image', metadata, rich_description, embedding, img_hash))

        if image_files.index(img_name) % 50 == 0:
            conn.commit()
            print(f"Progress: {image_files.index(img_name)}/{len(image_files)} images...")

    except Exception as e:
        print(f"Error on {img_name}: {e}")
        conn.rollback()

conn.commit()
print("\nSuccess! Your database is now medically literate.")
