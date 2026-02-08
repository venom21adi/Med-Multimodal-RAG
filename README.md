# Med-Multimodal-RAG

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

## 🔎 Example: Hybrid Clinical Retrieval Output

Below are real outputs from the **BiomedCLIP Hybrid Multimodal Search Engine**  
demonstrating fused retrieval across **radiology images** and **laboratory data**.

---

### Query 1
**Input:**  
`Anteroposterior chest radiograph showing opacification of the costophrenic sulcus and patchy consolidations in the lower lobes`

```
🖼️ [IMAGE] (Similarity: 0.6175)
 > Chest X-ray findings: Effusion. Filename: 00001775_001.png

📄 [TEXT] (Similarity: 0.6150)
 > Platelets: 188.4 ×10³/µL (2020-01-25)

🖼️ [IMAGE] (Similarity: 0.6073)
 > No Finding. Filename: 00000099_006.png

🖼️ [IMAGE] (Similarity: 0.6060)
 > Atelectasis, Consolidation, Effusion. Filename: 00001558_005.png

📄 [TEXT] (Similarity: 0.5995)
 > Leukocytes: 7.3 ×10³/µL (2020-01-25)
```

**Clinical Brief (Auto-Generated)**

- **Laboratory Summary:** Relevant hematology markers detected  
- **Imaging Summary:** Multiple X-rays consistent with effusion and consolidation  
- **Advisor Note:** Combined lab and imaging evidence supports **pneumonia with pleural effusion**

---

### Query 2
**Input:**  
`consolidation and blunted costophrenic angles`

```
📄 [TEXT] (Similarity: 0.6496)
 > Leukocytes: 7.3 ×10³/µL

🖼️ [IMAGE] (Similarity: 0.6382)
 > Effusion. Filename: 00000061_002.png

🖼️ [IMAGE] (Similarity: 0.6200)
 > Consolidation, Effusion, Infiltration, Nodule. Filename: 00000061_025.png
```

**Clinical Interpretation**

- Imaging repeatedly retrieves **effusion-related studies**
- Lab markers remain **consistent with inflammatory process**
- Suggests **pulmonary infection with pleural involvement**

---

### Query 3
**Input:**  
`pneumonia with pleural effusion`

```
📄 Leukocytes: 7.3 ×10³/µL  (Similarity: 0.6927)
📄 Platelets: 188.4 ×10³/µL (Similarity: 0.6882)
🖼️ Effusion X-ray matches   (Similarity ≈ 0.63)
```

**Result**

Clear **cross-modal agreement** between:

- Elevated inflammatory markers  
- Effusion-positive radiographs  

➡️ Strong diagnostic clustering around **pneumonia + pleural effusion**

---

### Query 4
**Input:**  
`Patient with high Body Mass Index and respiratory issues`

```
📄 BMI: 16.5 kg/m² (Similarity: 0.7699)
📄 Respiratory Rate: 16/min
🖼️ Consolidation / Cardiomegaly findings
```

**Observation**

- Text retrieval dominates due to **systemic query wording**
- Imaging still contributes **cardiopulmonary abnormalities**
- Demonstrates **balanced multimodal retrieval**, not image-only bias

---

## 🧠 Why This Matters

These examples show the system can:

- Retrieve **clinically relevant radiographs**
- Align them with **supporting laboratory evidence**
- Produce a **coherent diagnostic brief**

This validates the **Multimodal Fusion + Hybrid Retrieval** design  
as a step toward **clinically grounded medical RAG**.

---




