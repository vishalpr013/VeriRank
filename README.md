---
title: VeriRank
emoji: 🔍
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.10"
---

# VeriRank: Intelligent Candidate Discovery & Ranking System

This document outlines the architecture, setup instructions, design rationales, and execution commands for our **VeriRank** candidate ranking solution for the Redrob Data & AI Challenge. 

Our pipeline implements a **Pre-computed Dense Retrieval + Heuristic Rescoring Engine** designed to identify and rank the top 100 candidates for a **Founding Team Senior AI Engineer** role from a pool of 100,000 profiles.

---

## 1. Executive Summary & Rationale

*   **Execution Time**: Runs in **<8 seconds** on standard CPU (vs. 5-minute sandbox limit).
*   **RAM Footprint**: Consumes **<100 MB** of memory (vs. 16 GB sandbox limit).
*   **Safety & Compliance**: **100% network-independent** (no external API calls during ranking).
*   **Honeypot Protection**: **0% honeypot rate** in the final shortlist (passes the <10% disqualification threshold).
*   **Fact-Based Reasoning**: Programmatic generator matches candidate data perfectly with zero LLM-hallucination risk.

### Why We Designed It This Way
*   **Why not use live LLMs?** The sandbox runs completely offline, meaning we cannot access external APIs (OpenAI/Anthropic). Running a local LLM (like Llama-3) on CPU is extremely slow and will fail the 5-minute time constraint.
*   **Why pre-compute embeddings?** Generating vector representations for all 100K profiles on CPU takes ~50 minutes. By pre-filtering candidates down to a high-potential pool (28.8K candidates) and pre-computing their vectors, the ranker only runs high-speed `numpy` dot-products at execution time, decreasing retrieval latency to milliseconds.
*   **Why a deterministic reasoning generator?** Using an LLM to write candidate summaries risks hallucination (fabricating degrees/skills), which fails validation. Seeding a deterministic generator with the candidate's ID ensures 100% factual accuracy and high syntactic variety.

---

## 2. Pipeline Architecture

Our solution is divided into two distinct phases to achieve high accuracy within minimal compute times:

```
+--------------------------------------------------------------------------+
|                      OFFLINE PRE-COMPUTATION PHASE                       |
|                                                                          |
|  [candidates.jsonl] ---> [High-Recall Filter] ---> [28K Filtered Pool]   |
|                                                            |             |
|  [all-MiniLM-L6-v2] <--------------------------------------+             |
|         |                                                                |
|         v                                                                |
|  [candidate_embeddings.npz] & [model_weights/]                           |
+--------------------------------------------------------------------------+
                                     |
                                     v
+--------------------------------------------------------------------------+
|                      ONLINE RANKING STEP (rank.py)                       |
|                                                                          |
|  [candidate_embeddings.npz] ---> [Numpy Cosine Similarity]               |
|  [jd_embedding.npy]           --->          |                            |
|                                             v                            |
|  [candidates.jsonl]         ---> [Honeypot/Anomaly Filter]               |
|                                             |                            |
|                                             v                            |
|                                  [Heuristic Rescorer]                    |
|                                             |                            |
|                                             v                            |
|                                  [Top 100 Shortlist]                     |
|                                             |                            |
|                                             v                            |
|                                  [Reasoning Generator] ---> [Output CSV] |
+--------------------------------------------------------------------------+
```

### Phase A: Dense Retrieval (Semantic Search)
*   **Model**: **`all-MiniLM-L6-v2`** SentenceTransformer (384-dimensions, ~80 MB footprint).
*   **How it works**: Maps textual candidate profiles (concatenating current headline, summary, and skills list) and the Job Description into a vector space. Cosine similarity calculates the semantic distance between the JD requirements and the candidate.

### Phase B: Honeypot & Anomaly Filter
Identifies and drops anomalous profiles matching timeline traps:
*   *Expert Skills Trap*: Skips candidates claiming $\ge$ 5 "expert" skills with 0 months of use.
*   *Duration Mismatch*: Discards profiles where a job's start/end dates differ from its `duration_months` by $>3$ months.
*   *Pre-graduation career*: Discards candidates whose full-time careers started before they finished college.
*   *Twin roles*: Dropping profiles claiming more than 1 current full-time job.

### Phase C: Heuristic Rescorer & Sorter
*   **YoE Fit**: Rewards the ideal 5-9 years experience range.
*   **Location Fit**: Rewards local (Noida/Pune/Delhi) candidates or willingness to relocate.
*   **Industry Penalty**: Strongly penalizes service/consulting-only backgrounds (TCS, Infosys, Wipro, Accenture) and pure academic/research histories.
*   **Platform Activity**: Modifies score based on platform activity dates, response rates, GitHub activity, and university tiering.
*   **Tie-breaking**: Ranks are sorted descending by score, and ascending by `candidate_id` to guarantee strict lexicographical order.

---

## 3. File Structure

All project logic is self-contained inside the `Submission/` directory:

```
Submission/
├── README.md                     # Setup, design summary, and run instructions (This file)
├── understand.md                 # Technical design choices and rationale
├── requirements.txt              # Python dependencies (numpy, sentence-transformers, torch, tqdm, gradio, pandas)
├── app.py                        # Web UI interface script for Hugging Face Space
├── submission_metadata.yaml      # Submission metadata YAML
├── rank.py                       # Self-contained candidate ranking engine
├── generate_embeddings.py        # Offline standalone embedding builder script
├── validate_submission.py        # Submission validator script
├── redrob_output.csv              # Final validated ranked CSV output
└── vector_embeddings/            # Folder for pre-computed embeddings (generated at run)
    ├── candidate_embeddings.npz
    ├── jd_embedding.npy
    └── model_weights/

```

---

## 4. Installation & Setup

Install the required lightweight python libraries:
```bash
pip install -r requirements.txt
```

---

## 5. How to Run the Pipeline

The script `rank.py` is configured to run autonomously. If the pre-computed embedding files inside `vector_embeddings/` are missing, **it will automatically install any missing dependencies, download the model weights, pre-filter, and rebuild the embeddings from scratch before ranking.**

1.  Run the pipeline:
    ```bash
    python rank.py --candidates ../candidates.jsonl
    ```
2.  Validate the output file format:
    ```bash
    python validate_submission.py redrob_output.csv
    ```