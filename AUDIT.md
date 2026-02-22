
# 🔍 RAGAS Audit Report — Medical Multimodal RAG

This document summarizes structured evaluation results for the Medical Multimodal RAG pipeline using **RAGAS**.

---

## 🧠 Evaluation Configuration

- Generation Model: `mistral:7b-instruct` (Ollama)
- Judge Model: `mistral:7b-instruct`
- Embedding Model (RAGAS): `nomic-embed-text`
- Retrieval Embeddings: `BiomedCLIP`
- Vector Index: `PostgreSQL + pgvector (HNSW)`
- Query Rewriting: LLM-based clinical normalization (LOINC-style alignment)

Corpus Scale:
- ~152,000 structured Synthea clinical records
- ~3,500 NIH Chest X-ray studies

---

# 📊 Evaluation Runs

Each run below corresponds to a full RAGAS execution with structured JSON logs stored in `/ragas_audit/ragas_logs/`.

---

## 1️⃣ White Blood Cell Query

Original Query:
> Which patients have elevated white blood cell counts suggesting infection?

Rewritten Query:
> Leukocytes [#/volume] in Blood by Automated count

Source log: :contentReference[oaicite:0]{index=0}

### Results

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 0.750 |
| Answer Relevancy  | 0.855 |
| Context Precision | 1.000 |
| Context Recall    | 0.000 |

### Observation

- Retrieval was precise
- Retrieval coverage was insufficient
- Generation stayed grounded
- Demonstrates retrieval bottleneck

---

## 2️⃣ Resting Heart Rate Query

Original Query:
> Show me patients with high resting heart rate indicating possible tachycardia

Rewritten Query:
> Heart Rate [Rate/min] - Resting

Source log: :contentReference[oaicite:1]{index=1}

### Results

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 1.000 |
| Answer Relevancy  | 0.938 |
| Context Precision | 1.000 |
| Context Recall    | 0.500 |

### Observation

- High answer quality
- Partial retrieval coverage
- Example of generation masking incomplete retrieval

---

## 3️⃣ Blood Pressure Query

Original Query:
> What blood pressure readings are available for patients showing hypertension?

Rewritten Query:
> Systolic Blood Pressure, Diastolic Blood Pressure

Source log: :contentReference[oaicite:2]{index=2}

### Results

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 0.000 |
| Answer Relevancy  | 0.650 |
| Context Precision | 0.887 |
| Context Recall    | 0.500 |

### Observation

- Generation deviated from retrieved context
- Retrieval was moderately useful
- Indicates multi-variable failure (retrieval + grounding)

---

## 4️⃣ BMI Query

Original Query:
> Find patients with BMI in the underweight range

Rewritten Query:
> Body Mass Index

Source log: :contentReference[oaicite:3]{index=3}

### Results

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 1.000 |
| Answer Relevancy  | 0.903 |
| Context Precision | 1.000 |
| Context Recall    | 1.000 |

### Observation

- Full alignment across layers
- Retrieval and generation behaved optimally
- Demonstrates stable configuration case

---

## 5️⃣ Pain Intensity Query

Original Query:
> Which patients reported significant pain levels?

Rewritten Query:
> Pain, Self-Reported Intensity

Source log: :contentReference[oaicite:4]{index=4}

### Results

| Metric            | Score |
|-------------------|-------|
| Faithfulness      | 1.000 |
| Answer Relevancy  | 1.000 |
| Context Precision | 1.000 |
| Context Recall    | 0.000 |

### Observation

- Perfect generation metrics
- Zero retrieval recall
- Clear example of retrieval coverage failure masked by grounded answer

---

# 🧠 Key Insights

Across evaluation runs:

- High faithfulness does not guarantee sufficient retrieval coverage
- Context precision can be high while context recall is poor
- Query rewriting improves embedding alignment
- Retrieval depth and chunk strategy critically affect recall
- Generation stability can hide retrieval weaknesses

---

# 🎯 Conclusion

The RAGAS audit exposed layered failure modes within the Medical Multimodal RAG system.

The most critical discovery:

> Retrieval coverage, not prompt tuning, was the dominant reliability bottleneck.

This audit layer is now integrated as a first-class evaluation component in the pipeline.

---

# 📁 Logs

Structured JSON logs: ragas_logs
