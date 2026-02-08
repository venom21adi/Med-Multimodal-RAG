# Med-Multimodal-RAG

# 🏥 Med-Multimodal-RAG  
### Bridging the Clinical Modality Gap in Medical Retrieval

---

## 🎯 The Problem: The Clinical Modality Gap

Traditional **medical RAG systems** struggle with *true diagnostic relevance*.

When querying complex conditions like:

> “Pneumonia with Pleural Effusion”

standard **CLIP-based retrieval** often returns **“No Finding”** X-rays because:

- Models prioritize **visual similarity**
- Not **clinical correctness**
- Lack **systemic patient context**

**Result → Pretty images. Wrong diagnosis.**

This project fixes that.

---

## 🧠 The Core Idea

We build a **Multimodal Clinical Retrieval Engine** that aligns:

### 1️⃣ Anatomical Evidence
- Chest X-rays  
- Source: **NIH Chest X-ray14**

### 2️⃣ Systemic Evidence
- Blood lab measurements  
- Source: **Synthea synthetic EHR**
  - Leukocytes  
  - Platelets  
  - Other hematology markers  

Both modalities are embedded into a **shared clinical vector space**.

This transforms retrieval from:

**Image search → Diagnostic discovery**

---

## 🚀 The Secret Sauce: Fused Embedding Strategy

Most RAG pipelines embed **pixels only**.  
We **anchor images in clinical truth**.

### 1️⃣ Vector Fusion (Late Fusion)

During ingestion, we combine:

- Visual embedding from the X-ray  
- Expert-labeled clinical findings  

```
V_fused = Normalize(w1 * V_pixel + w2 * V_label)
```

### Why this matters

This **physically shifts** an image vector toward its:

- True **diagnostic neighborhood**
- Not just **visual look-alikes**

So:

- Effusion X-ray → retrieved for **Effusion query**  
- Not → random clean lung image

---

### 2️⃣ Hybrid Retrieval (Anti-Modality-Drowning)

We run **dual independent searches**:

- Top **N text records**
- Top **N image records**

Then merge and rank to produce a:

## 🩺 Clinical Brief

Guaranteeing:

- Radiology findings **present**
- Lab abnormalities **present**
- No single modality dominates retrieval

---

## 🏗️ Architecture

### Vector Database
- **PostgreSQL + pgvector**
- High-performance similarity search  
- Scalable and production-friendly

### Multimodal Model
- **BiomedCLIP**
  - ViT-B-16 (vision)
  - PubMedBERT (text)
- Pretrained on **biomedical corpora**

### Data Sources
- **NIH Chest X-ray14**
- **Synthea synthetic patient records**

---

## 📊 What This Enables

- Clinically meaningful **image retrieval**
- Multimodal **diagnostic context generation**
- Foundation for:
  - Clinical decision support
  - Radiology search engines
  - Medical AI copilots



## ⚠️ Disclaimer

This project is **for research purposes only**.  
**Not approved for clinical or diagnostic use.**

---



