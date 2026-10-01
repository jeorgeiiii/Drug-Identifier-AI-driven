<div align="center">

# 💊 Drug Identifier
### Giving approved drugs a second life with graph machine learning

![Graph ML](https://img.shields.io/badge/graph%20ML-DREAMwalk%20%2B%20TxGNN-4c1d95)
![Knowledge graph](https://img.shields.io/badge/knowledge%20graph-Hetionet%20v1.0-0e7490)
![Explainable](https://img.shields.io/badge/XAI-SHAP%20%7C%20saliency%20%7C%20ablation-be185d)
![Stack](https://img.shields.io/badge/stack-React%20%C2%B7%20Express%20%C2%B7%20FastAPI-1e293b)

**AUROC 0.9826** · 47k-node knowledge graph · two models, one verdict · every answer explained

</div>

---

## 🗺️ The project in one picture

![Knowledge map of the project generated with graphify](./docs/knowledge-map.png)

<sub>Generated with [graphify](https://github.com/safishamsi/graphify) from the thesis, poster and slides in this repo. Pink = TxGNN, blue = DREAMwalk, purple = the ensemble and web app where the two meet.</sub>

---

## 🤔 The question

> *"This drug is already approved for something. Could it also treat **that** disease?"*

Repurposing an approved drug skips years of safety testing, because the dosing, side effects and manufacturing are already known. But there are about **212,000** possible compound–disease pairs, and only **755** of them are known treatments. Nobody can test them all in a lab. This project narrows that huge space down to a short list worth testing, and shows its reasoning for every entry.

```mermaid
mindmap
  root((Why is this hard?))
    Almost no labels
      755 known treatments
      about 0.36% of all pairs
    Mixed data
      11 node types
      24 relation types
      29 source databases
    Black-box scores
      a number alone is not testable
    Hidden leakage
      test edges can sneak into embeddings
      and inflate metrics
```

---

## 🧪 How it works: following one prediction

Instead of listing components, let's follow a single question through the system:
**does Methotrexate treat Multiple Sclerosis?**

```mermaid
flowchart TD
    Q(["❓ Methotrexate → Multiple Sclerosis?"]):::q

    Q --> S1["<b>Step 1 · Look it up in the graph</b><br/>Both are nodes in Hetionet, surrounded by<br/>genes, pathways, side effects and anatomy"]
    S1 --> S2["<b>Step 2 · Hide the answer</b><br/>Known treat-edges in the test fold are deleted<br/>before any model sees the graph"]

    S2 --> W["<b>Step 3a · The explorer</b><br/>Random walkers wander the graph and sometimes<br/>teleport to similar drugs or diseases"]
    S2 --> M["<b>Step 3b · The gossip network</b><br/>Every node listens to its neighbours,<br/>relation by relation, for two rounds"]

    W --> W2["Walks → Skip-gram → 128-d vector per node<br/>pair = 256-d → XGBoost gives a probability"]
    M --> M2["Pretrain on all relations, fine-tune on 'treats'<br/>DistMult scores the pair"]

    W2 -->|"weight 0.55"| V{"<b>Step 4 · The vote</b>"}
    M2 -->|"weight 0.45"| V

    V --> R["<b>Step 5 · The receipt</b><br/>Why? → linked through genes IL1B and ALB<br/>TxGNN score 1.4152"]:::r

    classDef q fill:#fde68a,stroke:#b45309,color:#111
    classDef r fill:#bbf7d0,stroke:#15803d,color:#111
```

**Step 1: look it up.** Hetionet v1.0 is one big map of biomedicine: **47,031 nodes** and **2,250,197 edges** merged from 29 public databases. Methotrexate and Multiple Sclerosis are just two points on that map. Everything else around them is context the models can learn from.

**Step 2: hide the answer.** This step is easy to skip, and skipping it quietly breaks a lot of published pipelines. If the model could see the "Methotrexate treats X" edges it is being tested on, it would be cheating. So in **every fold**, the test treatment edges are removed *before* similarity, walks or embeddings are computed.

**Step 3a: the explorer (DREAMwalk + XGBoost).** Picture a walker hopping from node to node. Left alone, it would get stuck in the dense gene–gene network. So whenever it reaches a drug or disease, it can **teleport** to a semantically similar one, using similarity from vectorised Jaccard overlap and the ATC, MeSH and Disease Ontology hierarchies. The paths it takes are treated like sentences, and a type-aware **Skip-gram** turns them into 128-d vectors. A drug vector and a disease vector are joined into 256 numbers, and **XGBoost** (300 trees, depth 6) says how likely a treatment link is.

**Step 3b: the gossip network (TxGNN).** Here every node starts with a learnable 128-d vector and then *listens to its neighbours*. A gene hears from the pathways it belongs to, a drug hears from the genes it binds, and so on, with a separate `SAGEConv` for each relation type, stacked in two `HeteroConv` rounds. It first learns the whole graph, then specialises on "treats" edges. Negatives are drawn so that a real treatment is never mislabelled as a non-treatment. A **DistMult** decoder turns the two final vectors into a score.

**Step 4: the vote.** The two models learn in very different ways (where walkers end up vs. what neighbours say), so they make different mistakes. Their scores are blended **55 / 45**. A pair that both of them rank highly is a much safer bet.

**Step 5: the receipt.** A score is not enough for a scientist. The system also explains *why*. For Methotrexate → Multiple Sclerosis, the link runs through the genes **IL1B** and **ALB**. That is a hypothesis you can take into a lab.

---

## 🔍 Six ways to ask "why?"

```mermaid
flowchart LR
    P(("A ranked<br/>prediction"))
    P --> X1["🌲 TreeSHAP<br/><i>which features pushed it up?</i>"]
    P --> X2["🧬 Metapaths<br/><i>which genes & pathways link them?</i>"]
    P --> X3["📐 DistMult split<br/><i>which embedding dimensions?</i>"]
    P --> X4["🔦 Gradient saliency<br/><i>which neighbours mattered?</i>"]
    P --> X5["✂️ Relation ablation<br/><i>what breaks without a relation?</i>"]
    P --> X6["🗺️ Heatmaps<br/><i>patterns across many predictions</i>"]
```

The first two explain the XGBoost model and the last four explain TxGNN. Explanations use real names, never raw node IDs, and when two methods disagree you see both answers. One more example: **Etoposide → Breast Cancer** produces a SHAP force plot with **f(x) = 8.07**.

---

## 📊 What's inside the graph

```mermaid
pie showData title Hetionet v1.0: where the 47,031 nodes come from
    "Genes" : 20945
    "Pathways" : 1822
    "Compounds" : 1552
    "Diseases" : 137
    "Other types (anatomy, side effects, ...)" : 22575
```

Compounds and diseases, the two things we actually care about, make up only a small part of the graph. Most of what the models learn comes from everything *around* them.

---

## 🏁 Scorecard

```mermaid
xychart-beta
    title "Leak-free cross-validation (higher is better)"
    x-axis ["AUROC", "AUPR", "Accuracy"]
    y-axis "score" 0.80 --> 1.00
    bar [0.9826, 0.9869, 0.9342]
    bar [0.9396, 0.9378, 0.8775]
```

- 🟦 **TxGNN** (5-fold): AUROC **0.9826**, AUPR **0.9869**, accuracy **0.9342**. It was also compared against RGCN, DistMult and MLP baselines on the same harness.
- 🟩 **DREAMwalk + XGBoost** (10-fold): AUROC **0.9396**, AUPR **0.9378**, accuracy **0.8775**. That is almost identical to the original paper (0.938 / 0.939 / 0.873), even under the stricter leak-free protocol.
- 🧠 Full case studies on **Alzheimer's disease** and **breast carcinoma** are in the thesis.

---

## 🖥️ Using it

A researcher types a drug or disease into a **React** web app and picks a model (DREAMwalk, TxGNN or the ensemble). The request goes through a **Node.js / Express** gateway on port `8000`, which validates it, and then to a **FastAPI** inference service on port `8001`, which runs the models and the evidence pipeline. The researcher gets back a ranked list, the supporting evidence and an interactive graph. A built-in **XAI assistant** answers follow-up questions like *"why is this drug ranked first?"*

```mermaid
flowchart LR
    U["🧑‍🔬"] <--> FE["React SPA"] <--> GW["Express<br/>:8000"] <--> ML["FastAPI<br/>:8001"]
    ML <--> MOD[["DREAMwalk · TxGNN · Ensemble · Evidence"]]
```

---

## 🔭 What's next

```mermaid
mindmap
  root((Next steps))
    Evidence
      RAG over PubMed for each predicted link
      Neo4j enrichment from new literature
    Models
      Cross-graph tests on MSI and KEGG
      Multi-omics node features
      Attention-based ensemble weights
    Reality check
      In-vitro and in-vivo validation
    Product
      Hosted SaaS for researchers and clinicians
```

---

## 📚 Read more

- 📄 [**Full thesis**](./AI_Driven_Drug_Repurpose.pdf): methods, experiments and case studies
- 🖼️ [**Poster**](./AI_Drug_Repurposing_Poster%20.pdf): the whole project on one page
- 📊 [**Slides**](./AI-Driven%20Drug%20Repurposing%20final.pptx): presentation deck

## 🙌 Credits

Built by **Prince Mehra** for the Digilians 9-Month Diploma in Applied AI and Data Analytics (Military Technical College, August 2026).

<details>
<summary><b>References</b></summary>

1. Bang, D., Lim, S., Lee, S., & Kim, S. (2023). Biomedical knowledge graph learning for drug repurposing by extending guilt-by-association to multiple layers. _Nature Communications_.
2. Huang, K., & Zitnik, M. (2024). A foundation model for clinician-centered drug repurposing.
3. Zeng, X. et al. (2019). deepDR: a network-based deep learning approach to in silico drug repositioning.
4. Wang, Y. et al. (2022). DrugRepo: repurposing drugs based on chemical and genomic features.
5. Amiri, Razmara, Parvizpour & Izadkhah (2023). IDDI-DNN: drug repurposing via drug–disease association data integration using CNNs.
6. Talevi, A., & Bellera, C. (2020). Drug repurposing strategies, challenges, and opportunities.

</details>

<sub>⚠️ Predictions are computational hypotheses meant for lab follow-up. They are not medical advice.</sub>
