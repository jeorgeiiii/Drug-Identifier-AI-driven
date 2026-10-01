# AI-Driven Drug Repurposing with Knowledge Graphs & GNNs

![Project cover](./cover.png)

Finding a new disease for an already-approved drug is far cheaper and safer than developing a drug from scratch: the safety profile, dosing and manufacturing are already known. This project predicts such **drug → disease** links by learning from a large biomedical knowledge graph, and explains every prediction so a researcher can check *why* the model believes it.

Two independent models are trained on the same graph and their scores are blended:

- **DREAMwalk + XGBoost** — learns node embeddings from semantically guided random walks, then classifies drug–disease pairs.
- **TxGNN** — a heterogeneous graph neural network that learns through message passing across all relation types.

Because the two models learn in fundamentally different ways, a candidate both of them rank highly is far less likely to be a false positive.

📄 [Full thesis](./AI_Driven_Drug_Repurpose.pdf) · 🖼️ [Poster](./AI_Drug_Repurposing_Poster%20.pdf) · 📊 [Slides](./AI-Driven%20Drug%20Repurposing%20final.pptx)

---

## Contents

1. [The Problem](#the-problem)
2. [How It Works](#how-it-works)
3. [Data](#data)
4. [Model Details](#model-details)
5. [Explainability](#explainability)
6. [Results](#results)
7. [Application](#application)
8. [Ongoing & Future Work](#ongoing--future-work)
9. [Team](#team)
10. [References](#references)

---

## The Problem

| Challenge | What it means | How this project responds |
|---|---|---|
| **Sparse labels** | Only 755 of ~212k possible compound–disease pairs (≈0.36%) are known treatments | Learn from the *whole* graph (genes, pathways, side effects…), not just the labels |
| **Heterogeneous data** | 11 entity types and 24 relation types from 29 databases | Type-aware embeddings and relation-specific GNN layers |
| **Black-box models** | A score alone gives a scientist nothing to test | SHAP + graph-path explanations for every ranked candidate |
| **Hidden data leakage** | Test edges leaking into embeddings inflate metrics | Test-fold treatment edges are removed *before* any graph processing, every fold |

## How It Works

```
                Hetionet v1.0  (47,031 nodes · 2,250,197 edges)
                                   │
                 Leakage-free split: drop test-fold
                 Compound–treats–Disease edges first
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
   Branch A — DREAMwalk                         Branch B — TxGNN
   CSR sparse graph                             PyG HeteroData graph
   → vectorized Jaccard similarity              → 2-layer HeteroConv (SAGEConv)
   → semantic teleport random walks             → pretrain on all relations
   → heterogeneous Skip-gram (128-d)            → fine-tune on treats edges
   → XGBoost on 256-d pair vectors              → DistMult link scoring
            │                                             │
            └──────────────────────┬──────────────────────┘
                                   ▼
                 Ensemble score = 0.55·A + 0.45·B
                                   ▼
               Ranked candidates + explanations
```

## Data

[Hetionet v1.0](https://het.io/) integrates 29 public resources into one graph.

| | Count |
|---|---|
| Nodes / edges | 47,031 / 2,250,197 |
| Node types / relation types | 11 / 24 |
| Compounds · Diseases · Genes · Pathways | 1,552 · 137 · 20,945 · 1,822 |
| Known treatment (CtD) pairs | 755 |

## Model Details

**DREAMwalk + XGBoost**
- Random walks mix ordinary graph hops with *teleports* to semantically similar drugs or diseases, so dense gene–gene regions don't drown out the drug/disease signal.
- Similarity comes from vectorized Jaccard overlap and ontology information content (ATC, MeSH, Disease Ontology).
- A type-aware Skip-gram turns walks into 128-d embeddings; each pair becomes a 256-d vector.
- XGBoost classifier: 300 trees, depth 6, `hist` method. Evaluated with leak-free 10-fold stratified CV.

**TxGNN**
- 128-d learnable embeddings per node type, two `HeteroConv` layers with relation-specific `SAGEConv`, DistMult decoder.
- Two-phase training (all relations → treats-only), `BCEWithLogitsLoss`, negative sampling that never mislabels a known positive.
- Evaluated with leak-free 5-fold stratified CV and compared with RGCN, DistMult and MLP baselines under the same setup.

**Ensemble** — weighted average: 55% DREAMwalk/XGBoost, 45% TxGNN.

## Explainability

| Model | Method | Answers |
|---|---|---|
| XGBoost | TreeSHAP (summary, beeswarm, waterfall, force, dependence) | Which embedding features drove this score? |
| XGBoost | Metapath evidence | Which shared genes/pathways connect this drug and disease in Hetionet? |
| TxGNN | Exact DistMult decomposition | Which of the 128 dimensions contributed most? |
| TxGNN | Gradient neighbor saliency | Which real 1-hop neighbors mattered? |
| TxGNN | Leave-one-relation-out ablation | Which relation types does the prediction depend on? |
| TxGNN | Aggregated heatmaps | What patterns hold across many predictions? |

Explanations use real entity names (not node IDs), and disagreements between methods are reported rather than hidden.

**Examples:** Methotrexate → Multiple Sclerosis (explained through IL1B and ALB gene links); Etoposide → Breast Cancer (SHAP force plot, f(x) = 8.07).

## Results

| Metric | DREAMwalk + XGBoost (10-fold) | TxGNN (5-fold) |
|---|---|---|
| AUROC | 0.9396 | 0.9826 |
| AUPR | 0.9378 | 0.9869 |
| Accuracy | 0.8775 | 0.9342 |

The DREAMwalk reimplementation lands right next to the original paper's averages (0.938 / 0.939 / 0.873) despite the stricter leakage-free protocol. Case studies on **Alzheimer's disease** and **breast carcinoma** show the pipeline on real clinical questions.

## Application

| Tier | Stack | Responsibility |
|---|---|---|
| Frontend | React SPA | Search drugs/diseases, pick a model, browse ranked results, evidence and an interactive graph |
| Gateway | Node.js / Express (`:8000`) | Single API entry point; validation and response shaping |
| Inference | Python / FastAPI (`:8001`) | Serves DREAMwalk/XGBoost, TxGNN, the ensemble and the evidence pipeline |

An interactive XAI assistant lets users ask follow-up questions about any prediction.

## Ongoing & Future Work

- **RAG over PubMed** — retrieve published papers supporting each predicted link.
- **Live graph enrichment** — extend Hetionet in Neo4j with new drug–gene / gene–disease facts mined from recent literature.
- Cross-graph validation on MSI and KEGG; multi-omics node features; attention-based ensemble weighting.
- Lab (in-vitro / in-vivo) validation of top candidates.
- A hosted SaaS version for researchers and clinicians.

## Team

- Prince Mehra
- Gehan Abdulhameed
- Rahma Gamal
- Ruwayda Salah
- Manar Mohammed
- Bassant Adel
- Passant Mahmoud

Built for the Digilians 9-Month Diploma in Applied AI and Data Analytics (Military Technical College, August 2026).

## References

1. Bang, D., Lim, S., Lee, S., & Kim, S. (2023). Biomedical knowledge graph learning for drug repurposing by extending guilt-by-association to multiple layers. *Nature Communications*.
2. Huang, K., & Zitnik, M. (2024). A foundation model for clinician-centered drug repurposing.
3. Zeng, X. et al. (2019). deepDR: a network-based deep learning approach to in silico drug repositioning.
4. Wang, Y. et al. (2022). DrugRepo: repurposing drugs based on chemical and genomic features.
5. Amiri, Razmara, Parvizpour & Izadkhah (2023). IDDI-DNN: drug repurposing via drug–disease association data integration using CNNs.
6. Talevi, A., & Bellera, C. (2020). Drug repurposing strategies, challenges, and opportunities.

---

*Research project — predictions are computational hypotheses for experimental follow-up, not medical advice.*
