# Med-Multimodal-RAG

🏥 Med-Multimodal-RAG: Bridging the Clinical Modality Gap

🎯 The "Why": Solving the Modality GapTraditional Medical RAG systems often suffer from a Modality Gap. 
When searching for complex pathologies like "Pneumonia with Pleural Effusion," standard CLIP-based models frequently return "clean" (No Finding) images because they lack clinical context, prioritizing image composition over diagnostic truth.
This project implements a Multimodal Clinical Engine that correlates:Anatomical Evidence: Chest X-rays (NIH Dataset).Systemic Evidence: Laboratory blood work (Synthea-generated Leukocytes, Platelets, etc.).
By aligning these two distinct data streams into a single vector space, we move beyond simple "image search" into Diagnostic Discovery.

🚀 The "How": Fused Embedding Strategy (The Secret Sauce)Most RAG implementations treat images as raw pixels. This repo uses a Multimodal Fusion Ingestion Pipeline to "anchor" visual data in clinical truth.
1. Vector Fusion (Late Fusion)During ingestion, we don't just embed the image. We perform a weighted blend of the visual features and the expert-labeled clinical findings:Vfused​=Normalize(w1​⋅Vpixel​+w2​⋅Vlabel​)
This physically shifts the image vector closer to its diagnostic neighborhood, ensuring that an "Effusion" X-ray actually ranks for "Effusion" queries.

2. Hybrid RetrievalThe system utilizes a dual-stream retrieval process to prevent "Modality Drowning." 
By fetching the top N text records and top $N$ image records independently before ranking, we ensure the final "Clinical Brief" contains both lab results and radiology findings.

🏗️ ArchitectureVector Database: PostgreSQL with pgvector for high-performance similarity search.
Model: BiomedCLIP (ViT-B-16 + PubMedBERT), specifically pre-trained on medical domain data.
Data Sources: NIH Chest X-ray 14 & Synthea patient records.
