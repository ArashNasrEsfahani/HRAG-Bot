# Phase 2 v2 Benchmark Results

Run timestamp: 2026-05-15T19:54:43+00:00
Total wall time: 667.0s
Pass rate: 4/9 questions

## Config snapshot

| Setting | Value |
|---|---|
| retriever | router |
| reranker | cross_encoder |
| top_k_vector | 20 |
| top_k_final | 15 |
| rerank_threshold | -8.0 |
| kg.enabled | True |
| kg.use_communities | True |
| llm.model | gemma4:e4b |

## Summary table

| id | category | overall | substrings | match_tier | citation | raft | src match | no fallback | rewriter | no halluc | route match | route classified | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | factual_single_source | FAIL | FAIL | - | PASS | PASS | PASS | PASS | PASS | PASS | PASS | entity | 60.7 |
| q2 | factual_single_source | FAIL | PASS | exact | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | cross_document | 51.8 |
| q3 | factual_single_source | PASS | PASS | exact | PASS | PASS | PASS | PASS | PASS | PASS | PASS | entity | 67.8 |
| q4 | multi_hop | PASS | PASS | exact | PASS | PASS | PASS | PASS | PASS | PASS | PASS | entity | 50.2 |
| q5 | cross_document | FAIL | FAIL | - | PASS | PASS | PASS | PASS | PASS | PASS | PASS | cross_document | 49.2 |
| q6 | cross_document | PASS | PASS | exact | PASS | PASS | PASS | PASS | PASS | PASS | PASS | cross_document | 56.0 |
| q7 | conversational_followup | FAIL | FAIL | - | PASS | PASS | PASS | PASS | PASS | PASS | PASS | entity | 144.4 |
| q8 | out_of_corpus | FAIL | PASS | exact | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | entity | 36.8 |
| q9 | numeric_fact | PASS | PASS | exact | PASS | PASS | PASS | PASS | PASS | PASS | PASS | entity | 52.3 |

## Per-question detail

### q1 — factual_single_source

**Question:** What two multi-hop QA benchmarks does HippoRAG primarily evaluate on, and by what margin does it outperform baselines on the harder of the two?
**Answer (excerpt, first 400 chars):** Reasoning: The user asks for two multi-hop QA benchmarks that HippoRAG primarily evaluates on, and the performance margin on the harder of the two.  I scanned the passages for mentions of "HippoRAG" and quantitative comparisons on multiple benchmarks.  [Source 3] provides the most specific answer: "Notably, HippoRAG 2 outperforms NV-Embed-v2 by 9.5% F1 on 2Wiki and by 3.1% on the challenging LV-Ev

**Sources returned:**
  - [1] 4043_ReMindRAG_Low_Cost_LLM_Gu / 5.2.1 (retriever=router)
  - [2] 2502.14802v2 / Results (retriever=router)
  - [3] 2502.14802v2 / 78.2 (retriever=router)
  - [4] 2512.10422v3 / 20,686 (retriever=router)
  - [5] 2502.14802v2 / Discourse understanding evaluates sense-making by (retriever=router)
  - [6] 2502.14802v2 / 74.7 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** entity
**Expected route:** entity
**Route match:** PASS
**KG path hit:** yes

**Match tier:** -
  - `MuSiQue`: FAIL (tier=fuzzy)
  - `2WikiMultiHopQA`: FAIL (tier=exact)
  - `20`: FAIL (tier=exact)


**Missing substrings:** ['MuSiQue', '2WikiMultiHopQA', '20']

**Dimension scores:**
- substrings: FAIL
- match_tier: -
- judge_used: True
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): entity
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** HIPPORAG.pdf abstract (page 1) states HippoRAG is evaluated on MuSiQue and 2WikiMultiHopQA, achieving "strong performance improvements of around 3 and 20 points over current RAG methods". Both benchmark names are distinctive proper nouns that appear only in HIPPORAG.pdf at this specificity level. "20" matches the margin figure and is robust because the sentence pairs it with 2WikiMultiHopQA — the LLM must retrieve the abstract/intro chunk. Neither benchmark name appears in 2025.findings-naacl.30.pdf. NEW question (not in phase1 or phase2) — chosen to exercise the entity path on HIPPORAG with a factual question not reused from q4.

---

### q2 — factual_single_source

**Question:** What dataset does RAGate use for training and evaluation, and how many dialogues does it contain?
**Answer (excerpt, first 400 chars):** Reasoning: The user is asking for the name of the dataset used for RAGate's training and evaluation, and the number of dialogues it contains.  1.  I scanned the passages for mentions of RAGate and associated datasets. 2.  [Source 1] and [Source 2] mention that RAGate is evaluated on the "KETOD dataset." 3.  [Source 11] confirms this, stating that the classification performance is evaluated on the 

**Sources returned:**
  - [1] 2025.findings-naacl.30 / 5.3 (retriever=router)
  - [2] 2025.findings-naacl.30 / 9.3425 (retriever=router)
  - [3] 2025.findings-naacl.30 / 5.1 (retriever=router)
  - [4] 2025.findings-naacl.30 / 5.1 (retriever=router)
  - [5] 2025.findings-naacl.30 / (2) (retriever=router)
  - [6] 2025.findings-naacl.30 / 5.2 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** cross_document
**Expected route:** entity
**Route match:** FAIL
**KG path hit:** yes

**Match tier:** exact


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- match_tier: exact
- judge_used: False
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: FAIL
- route_classified (v2): cross_document
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** Verbatim carry-over from phase1 q2, which PASSED. 2025.findings-naacl.30.pdf pages 4-5 (Section 4 Model Training and Evaluation Setups): "we use the KETOD dataset (Chen et al., 2022), which has fully annotated 5,324 dialogues and 52,063 turns of conversations". KETOD is a distinctive dataset name absent from HIPPORAG.pdf and 2502.14802v2.pdf. "5,324" is a precise count that cannot appear by coincidence. Tests that the entity path independently retrieves from the RAGate paper, confirming per-paper retrievability under the patched router.

---

### q3 — factual_single_source

**Question:** What is the default method HippoRAG 2 uses to link queries to the knowledge graph, and which embedding model does it adopt as its primary retriever?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for two specific pieces of information regarding HippoRAG 2: its default query linking method and its primary retriever embedding model.  1.  **Identifying the Default Linking Method:** Source 1 directly addresses this, stating that HippoRAG 2's approaches for linking queries to the KG include NER to Node, Query to Node, and Query to Triple, and specifies: "By default,

**Sources returned:**
  - [1] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [2] 2502.14802v2 / Related Work (retriever=router)
  - [3] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [4] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [5] 2502.14802v2 / 78.2 (retriever=router)
  - [6] 2502.01113v3 / 3.2 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** entity
**Expected route:** entity
**Route match:** PASS
**KG path hit:** yes

**Match tier:** exact


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- match_tier: exact
- judge_used: False
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): entity
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: PASS**

**Notes:** NEW question targeting 2502.14802v2 exclusively. Section 3.3 of 2502.14802v2.pdf (Deeper Contextualization) states "By default, HippoRAG 2 adopts the query-to-triple approach". Section 4.1 (Baselines) and Section 4.4 (Implementation Details) name "nvidia/NV-Embed-v2" as the primary retriever. Both phrases are verbatim and unique to 2502.14802v2 — "query-to-triple" does not appear in HIPPORAG.pdf (which uses NER-to-node) and "NV-Embed-v2" only appears as a baseline name in that paper. Tests that the entity path retrieves from HippoRAG 2 independently, confirming all three papers are reachable under the patched router.

---

### q4 — multi_hop

**Question:** In HippoRAG, what algorithm is used during online retrieval, and what two hyperparameters were tuned using MuSiQue training data?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for two pieces of information: the algorithm used during online retrieval in HippoRAG, and the two hyperparameters tuned using MuSiQue training data.  1.  **Finding the Online Retrieval Algorithm:** Multiple passages confirm the algorithm. Source 1 states that for online retrieval, seed nodes are used for the "Personalized PageRank (PPR) algorithm" [Source 1]. Source 2

**Sources returned:**
  - [1] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [2] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [3] 2512.10422v3 / 20,686 (retriever=router)
  - [4] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [5] HIPPORAG / RAG (retriever=router)
  - [6] 2512.10422v3 / 4.7 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** entity
**Expected route:** entity
**Route match:** PASS
**KG path hit:** yes

**Match tier:** exact


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- match_tier: exact
- judge_used: False
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): entity
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: PASS**

**Notes:** Verbatim carry-over from phase1 q3 and phase2 q3, both of which PASSED. HIPPORAG.pdf Section 2.2 (page 3) names Personalized PageRank as the online retrieval algorithm. Section 3.4 (page 6) states "the synonymy threshold τ at 0.8 and the PPR damping factor at 0.5" as the two hyperparameters tuned on MuSiQue training data. No single chunk answers both sub-questions — the answer requires combining two passages, which is exactly the multi-hop scenario KG-PPR graph traversal is designed for. Re-using this question provides a clean PASS-vs-PASS baseline diff between phase1, phase2, and phase2_v2.

---

### q5 — cross_document

**Question:** Both HippoRAG and RAGate address a limitation of standard RAG. What limitation does HippoRAG address, and what limitation does RAGate address?
**Answer (excerpt, first 400 chars):** Reasoning: The user asks for the specific limitations addressed by HippoRAG and RAGate, relative to standard RAG.  1.  **Finding the HippoRAG limitation:** Source 6 directly addresses this. It states that the original method used in HippoRAG (NER) is "predominantly concept-centric, often overlooking the contextual alignment within the KG" [Source 6]. The subsequent methods (Query to Node, Query to

**Sources returned:**
  - [1] 2025.findings-naacl.30 / 5.2 (retriever=router)
  - [2] 2025.findings-naacl.30 / 5.1 (retriever=router)
  - [3] 2025.findings-naacl.30 / 5.3 (retriever=router)
  - [4] 2025.findings-naacl.30 / 9.3425 (retriever=router)
  - [5] 2025.findings-naacl.30 / 5.1 (retriever=router)
  - [6] 2502.14802v2 / HippoRAG 2 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** cross_document
**Expected route:** cross_document
**Route match:** PASS
**KG path hit:** yes

**Match tier:** -
  - `knowledge integration`: FAIL (tier=fuzzy)
  - `every turn`: FAIL (tier=fuzzy)


**Missing substrings:** ['knowledge integration', 'every turn']

**Dimension scores:**
- substrings: FAIL
- match_tier: -
- judge_used: True
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): cross_document
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** Verbatim carry-over from phase1 q4 and phase2 q4 — the primary acceptance criterion for Phase 2. HIPPORAG.pdf Figure 1 caption reads "Knowledge Integration & RAG" and the introduction states the goal of "knowledge integration over new experiences". 2025.findings-naacl.30.pdf introduction (page 2) states "the necessity of augmenting every turn of the conversation with external knowledge remains questionable". "knowledge integration" anchors to HippoRAG; "every turn" anchors to RAGate. Both must appear in the answer, requiring the router to take the cross_document path and fuse chunks from both papers via RRF. FAIL→PASS here remains the primary go/no-go signal for the Phase 2 router fix.

---

### q6 — cross_document

**Question:** Both HippoRAG and HippoRAG 2 use the Personalized PageRank algorithm. What does the original HippoRAG use as seed nodes for PPR, and what additional node type does HippoRAG 2 also include as seed nodes?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for the seed nodes used by both HippoRAG and HippoRAG 2 for the Personalized PageRank (PPR) algorithm.  1.  **Identifying HippoRAG's seed nodes:**     *   Source 1 states that in HippoRAG's online retrieval process, the LLM neocortex extracts named entities from a query, and the nodes corresponding to these entities are used as seed nodes for PPR.     *   Source 2 corr

**Sources returned:**
  - [1] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [2] HIPPORAG / RAG (retriever=router)
  - [3] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [4] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [5] HIPPORAG / LLM (retriever=router)
  - [6] HIPPORAG / 2.3 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** cross_document
**Expected route:** cross_document
**Route match:** PASS
**KG path hit:** yes

**Match tier:** exact


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- match_tier: exact
- judge_used: False
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): cross_document
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: PASS**

**Notes:** NEW cross-document question forcing retrieval from two closely related papers. HIPPORAG.pdf Section 2.3 (page 4) states that "named entities from the query" (query named entities Cq) become the seed nodes for PPR — the phrase "named entities" is the key anchor. 2502.14802v2.pdf Section 3.5 (Online Retrieval) states "All passage nodes are also taken as seed nodes". The two papers share vocabulary (PPR, seeds) but differ on seed node types — a correct answer must cite both. High lexical overlap between HIPPORAG and HippoRAG 2 means a simple vector retriever easily confines results to one paper; the cross_document RRF path is required to surface both. Second cross-doc case prevents a single fluky router classification from masking a real failure.

---

### q7 — conversational_followup

**Turn 1:** What is RAGate and what problem does it solve?
**Turn 2:** What are the three variants it uses to implement the gating function?
**Answer (excerpt, first 400 chars):** [T1] Reasoning: The user is asking for a definition and the problem solved by RAGate. Multiple passages discuss RAGate's function and benefits.  1.  **Definition/Function:** [Source 3] identifies RAGate for Response Generation and mentions using RAGate-PEFT and RAGate-MHA to support "adaptive retrieval augmented conversational response generation." [Source 4] further discusses implementing the RAG

**Sources returned:**
  - [1] 2025.findings-naacl.30 / 5.3 (retriever=router)
  - [2] 2025.findings-naacl.30 / 3.2 (retriever=router)
  - [3] 2025.findings-naacl.30 / 9.3425 (retriever=router)
  - [4] 2025.findings-naacl.30 / 5.1 (retriever=router)
  - [5] 2025.findings-naacl.30 / 9.3425 (retriever=router)
  - [6] 2025.findings-naacl.30 / 5.1 (retriever=router)

**Route distribution (source stamps):** {'router': 24}
**Dominant route (source stamps):** router
**Route classified (router event):** entity
**Expected route:** N/A
**Route match:** PASS
**KG path hit:** yes

**Match tier:** -
  - `RAGate`: PASS (tier=exact)
  - `gating`: FAIL (tier=fuzzy)
  - `external knowledge`: PASS (tier=exact)
  - `RAGate-Prompt`: FAIL (tier=fuzzy)
  - `RAGate-PEFT`: PASS (tier=exact)
  - `RAGate-MHA`: PASS (tier=exact)

**Query rewrite:** `What are the three variants it uses to implement the gating function?` → `What is RAGate and what problem does it solve?
What are the three variants it uses to implement the gating function?`

**Missing substrings:** ['T1:RAGate', 'T1:gating', 'T1:external knowledge', 'T2:RAGate-Prompt']

**Dimension scores:**
- substrings: FAIL
- match_tier: -
- judge_used: True
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): entity
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** Verbatim carry-over from phase1 q5 and phase2 q5, both of which PASSED. Turn 1 is self-contained: answered by 2025.findings-naacl.30.pdf pages 1-2 (RAGate is a gating model that adaptively determines when to augment with external knowledge). Turn 2 uses "it" and "the three variants" without restating the subject, requiring the heuristic query rewriter to expand to "What are the three variants RAGate uses to implement the gating function?" before hitting the router. 2025.findings-naacl.30.pdf Section 3.2 Figure 2 caption and body name all three: RAGate-Prompt (LLM prompting), RAGate-PEFT (parameter-efficient fine-tuning), and RAGate-MHA (multi-head attention encoder). These three hyphenated names are unique to this paper. expected_route is null because the turn-1 router path is incidental; turn-2 entity routing is the exercised behaviour.

---

### q8 — out_of_corpus

**Question:** What is the date of birth of Bernal Jiménez Gutiérrez, the lead author of HippoRAG?
**Answer (excerpt, first 400 chars):** Reasoning: I searched all retrieved passages for the date of birth of Bernal Jiménez Gutiérrez. Source 1 identifies Bernal Jiménez Gutiérrez as an author and lists his affiliation as "The Ohio State University," but it does not provide a date of birth. Source 7 also mentions "Bernal Jim´enez Gutiérrez" and provides his email address and affiliation, but it does not contain any date of birth. There

**Sources returned:**
  - [1] HIPPORAG / N/A (retriever=router)
  - [2] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [3] 2502.14802v2 / 74.7 (retriever=router)
  - [4] 2502.14802v2 / 6.6 (retriever=router)
  - [5] 2502.14802v2 / 65.6 (retriever=router)
  - [6] HIPPORAG / 52.0 (retriever=router)

**Route distribution (source stamps):** {'router': 7}
**Dominant route (source stamps):** router
**Route classified (router event):** entity
**Expected route:** N/A
**Route match:** PASS
**KG path hit:** yes

**Match tier:** exact


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- match_tier: exact
- judge_used: False
- citation: FAIL
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): entity
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** NEW out-of-corpus question replacing phase2 q6 (BLEU/GPT-4o), which was too plausible and caused hallucination because HippoRAG 2 NarrativeQA F1 passages were retrieved and cited. An author's date of birth is biographical metadata that no research paper contains — it is clearly out-of-domain (personal facts vs. technical content) and cannot be inferred from anything in the corpus. "Bernal Jiménez Gutiérrez" is the real lead author (confirmed from HIPPORAG.pdf and 2502.14802v2.pdf author lists) so the name is plausible and prevents the system from refusing on the name alone, but the fact itself is strictly absent. Tests the "couldn't find" fallback path cleanly without citation interference.

---

### q9 — numeric_fact

**Question:** What default weight factor does HippoRAG 2 assign to passage nodes when initialising PPR reset probabilities, according to its ablation study?
**Answer (excerpt, first 400 chars):** Reasoning: The user is asking for the default weight factor used by HippoRAG 2 for passage nodes when initializing the Personalized PageRank (PPR) reset probabilities. Source 2, titled "6.2. Controlling Reset Probabilities," directly addresses this topic. It explains that balancing the reset probabilities between phrase nodes and passage nodes is necessary during PPR [Source 2]. The passage specif

**Sources returned:**
  - [1] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [2] 2502.14802v2 / Discussions (retriever=router)
  - [3] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [4] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [5] 2502.14802v2 / 86.4 (retriever=router)
  - [6] 2502.14802v2 / 74.7 (retriever=router)

**Route distribution (source stamps):** {'router': 12}
**Dominant route (source stamps):** router
**Route classified (router event):** entity
**Expected route:** entity
**Route match:** PASS
**KG path hit:** yes

**Match tier:** exact


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- match_tier: exact
- judge_used: False
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_match: PASS
- route_classified (v2): entity
- route_observed (info, dominant): router
- kg_path_hit (info): yes

**Score: PASS**

**Notes:** NEW numeric question targeting 2502.14802v2 exclusively. Section 6.2 and Table 5 of 2502.14802v2.pdf (Controlling Reset Probabilities) state: "we set the factor to be 0.05 by default" for passage node reset probabilities in PPR. "0.05" is an exact floating-point string that cannot appear by coincidence and uniquely identifies this specific experimental finding. "passage node" is the precise technical term introduced in Section 3.2 of the same paper. The combination distinguishes this chunk from any other numeric mention. Tests that the entity path can retrieve a specific implementation detail from the ablation section of HippoRAG 2, exercising precision retrieval beyond the abstract/intro.

---
