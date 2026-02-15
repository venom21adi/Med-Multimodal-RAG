## 🚀 HNSW Performance Benchmark

To evaluate scalability, we indexed:

- **~150,000 multimodal clinical records**
- **~2,500 chest X-ray images**
- Joint embeddings using **BiomedCLIP**
- Retrieval via **PostgreSQL + pgvector (HNSW index)**

---

### ⏱️ Latency

Across repeated clinical queries:

- **Execution time:** ~0.18 – 0.43 seconds  
- **Stable across iterations**
- Suitable for **interactive clinical retrieval**

---

### 🎯 Similarity Quality

Top-match cosine similarity remained:

- **0.65 – 0.78 range**
- Consistent across:
  - Pneumonia with pleural effusion  
  - Consolidation with blunted costophrenic angles  
  - Respiratory distress with leukocytosis  
  - BMI-related respiratory cases  

This indicates **stable semantic neighborhood retrieval** at scale.

---

### 🩺 Cross-Modal Clinical Consistency

For representative query:

**“pneumonia with pleural effusion”**

The system retrieved:

- Relevant **laboratory markers**
- Effusion-positive **radiographs**
- Auto-generated **clinical brief** aligning both modalities

➡️ Demonstrates **diagnostic agreement across modalities**  
➡️ Achieved at **sub-second latency**

---

## 📊 Key Takeaway

This benchmark validates that:

**Multimodal clinical RAG can be both:**

- **Clinically meaningful**
- **Operationally scalable**

— a necessary step toward **real-world healthcare AI systems**.
