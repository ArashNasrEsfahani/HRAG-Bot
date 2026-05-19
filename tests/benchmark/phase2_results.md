# Phase 2 Benchmark Results

Run timestamp: 2026-05-09T12:15:32+00:00
Total wall time: 375.7s
Pass rate: 0/5 questions

## Config snapshot

| Setting | Value |
|---|---|
| retriever | router |
| reranker | cross_encoder |
| top_k_vector | 20 |
| top_k_final | 10 |
| rerank_threshold | -8.0 |
| kg.enabled | True |
| kg.use_communities | False |
| llm.model | gemma4:e4b |

## Summary table

| id | category | overall | substrings | citation | raft | src match | no fallback | rewriter | no halluc | route | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| q1 | factual_single_source | FAIL | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | router | 103.4 |
| q3 | multi_hop | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | router | 47.3 |
| q4 | cross_document | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | router | 74.4 |
| q5 | conversational_followup | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | router | 102.3 |
| q6 | out_of_corpus | FAIL | FAIL | FAIL | FAIL | PASS | PASS | PASS | FAIL | router | 48.3 |

## Per-question detail

### q1 — factual_single_source

**Question:** What improvement does HippoRAG 2 achieve over standard RAG in associativity tasks, and what retriever does it use by default?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for two pieces of information regarding HippoRAG 2: its improvement in associativity tasks and its default retriever.  1.  **Associativity/Improvement:**     *   [Source 1] defines associativity as the capacity to draw multi-hop connections and notes that HippoRAG 2 is closely inspired by HippoRAG and "allows it to perform very well on multi-hop QA tasks. However, its 

**Sources returned:**
  - [1] 2502.14802v2 / Related Work (retriever=router)
  - [2] 2502.14802v2 / 78.2 (retriever=router)
  - [3] 2502.14802v2 / 74.7 (retriever=router)
  - [4] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [5] 121_Memory_Never_Fades_Boostin / 45.2 (retriever=router)
  - [6] 2409.05591v3 / 3.3 (retriever=router)

**Route distribution:** {'router': 10}
**Dominant route:** router
**KG path hit:** yes


**Missing substrings:** ['7', 'NV-Embed-v2']

**Dimension scores:**
- substrings: FAIL
- citation: PASS
- raft_blocks: FAIL
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_observed (info): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** HippoRAG 2 abstract (2502.14802v2.pdf, page 1) states "achieving a 7% improvement in associative memory tasks over the state-of-the-art embedding model". Section 4.4 (page 6) names "nvidia/NV-Embed-v2 as the retriever". Both facts appear only in 2502.14802v2, not in HIPPORAG.pdf or 2025.findings-naacl.30.pdf. The number "7" combined with "associativity" is a precise discriminator. This question is routed as entity/factual — exercises basic KG-PPR seed-finding without requiring cross-doc synthesis, confirming Phase 1 baseline is not broken by the router.

---

### q3 — multi_hop

**Question:** In HippoRAG, what algorithm is used during online retrieval, and what two hyperparameters were tuned using MuSiQue training data?
**Answer (excerpt, first 400 chars):** Reasoning: The user asks for two pieces of information: the algorithm used during online retrieval in HippoRAG, and two hyperparameters tuned using MuSiQue training data.  1.  **Algorithm:** Source 1 describes the online retrieval process. It states that after selecting seed nodes and assigning reset probabilities, "The PPR search is then executed, and passages are ranked by their PageRank scores,

**Sources returned:**
  - [1] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [2] 2512.10422v3 / 20,686 (retriever=router)
  - [3] HIPPORAG / 2.2 (retriever=router)
  - [4] 2502.14802v2 / Discussions (retriever=router)
  - [5] 2502.01113v3 / 4.1 (retriever=router)
  - [6] 2512.10422v3 / 4.7 (retriever=router)

**Route distribution:** {'router': 10}
**Dominant route:** router
**KG path hit:** yes


**Missing substrings:** ['synonymy threshold', 'damping factor']

**Dimension scores:**
- substrings: FAIL
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_observed (info): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** Verbatim carry-over from phase1 q3, which PASSED. HIPPORAG.pdf Section 2.2 (page 3) names Personalized PageRank as the online retrieval algorithm; Section 3.4 (page 6) states "the synonymy threshold τ at 0.8 and the PPR damping factor at 0.5" as the two hyperparameters tuned on MuSiQue. Neither chunk alone answers both parts, so the answer requires combining two passages — exactly what KG-PPR graph traversal is designed for. Re-using this question preserves a clean PASS-vs-PASS diff between phase1 and phase2 on the multi_hop dimension.

---

### q4 — cross_document

**Question:** Both HippoRAG and RAGate address a limitation of standard RAG. What limitation does HippoRAG address, and what limitation does RAGate address?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for the limitations addressed by both HippoRAG and RAGate.  For HippoRAG, Source 1 is highly relevant. It defines "Associativity" as the capacity to draw multi-hop connections between disparate facts for efficient retrieval, noting that "standard RAG cannot emulate due to its reliance on independent vector retrieval" [Source 1]. It then states that "HippoRAG (Guti´erre

**Sources returned:**
  - [1] 2502.14802v2 / Related Work (retriever=router)
  - [2] 121_Memory_Never_Fades_Boostin / 3.6 (retriever=router)
  - [3] 121_Memory_Never_Fades_Boostin / 3.5 (retriever=router)
  - [4] 2501.14342v3 / Avg. Tokens (retriever=router)
  - [5] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [6] 2406.14497v2 / RAG (retriever=router)

**Route distribution:** {'router': 10}
**Dominant route:** router
**KG path hit:** yes


**Missing substrings:** ['knowledge integration', 'every turn']

**Dimension scores:**
- substrings: FAIL
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_observed (info): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** Verbatim copy of phase1 q4 — the primary acceptance criterion for Phase 2. HIPPORAG.pdf abstract and introduction (pages 1-2) describe HippoRAG's goal as enabling "knowledge integration over new experiences" — the phrase "knowledge integration" is used in the figure caption (Figure 1 title: "Knowledge Integration & RAG") and in the body text. 2025.findings-naacl.30.pdf introduction (page 2) states "the necessity of augmenting every turn of the conversation with external knowledge remains questionable" — "every turn" is the diagnostic phrase. Phase 1 FAILED this question (substrings dimension) because the hybrid retriever returned only HippoRAG 2 chunks, missing the RAGate paper entirely. The LLM router's cross_document path fuses results from both papers via RRF, which should surface the RAGate chunk containing "every turn". FAIL→PASS here is the go/no-go signal for Phase 2.

---

### q5 — conversational_followup

**Turn 1:** What is RAGate and what problem does it solve?
**Turn 2:** What are the three variants it uses to implement the gating function?
**Answer (excerpt, first 400 chars):** [T1] Reasoning: The user is asking for a definition of "RAGate" and the problem it solves. I searched the retrieved passages for this term. Source 5 and Source 7 contain the phrase "RAGate-MHA" and describe its input formats: *   [Source 5] mentions "RAGate-MHA: Context-Response Input". *   [Source 7] mentions "RAGate-MHA: Context with / without Knowledge Input".  While these passages identify the

**Sources returned:**
  - [1] 2501.15228v2 / 4.3 (retriever=router)
  - [2] 2501.15228v2 / 3.2 (retriever=router)
  - [3] 2501.15228v2 / MMOA-RAG (retriever=router)
  - [4] 2404.10981v2 / 3.1 (retriever=router)
  - [5] 2404.10981v2 / 20: (retriever=router)
  - [6] 2025.findings-naacl.30 / 0.3271 (retriever=router)

**Route distribution:** {'router': 20}
**Dominant route:** router
**KG path hit:** yes

**Query rewrite:** `What are the three variants it uses to implement the gating function?` → `What is RAGate and what problem does it solve?
What are the three variants it uses to implement the gating function?`

**Missing substrings:** ['T1:gating', 'T1:external knowledge', 'T2:RAGate-Prompt', 'T2:RAGate-PEFT']

**Dimension scores:**
- substrings: FAIL
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS
- route_observed (info): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** Verbatim carry-over from phase1 q5, which PASSED. Turn 1 asks a self-contained factual question answered by 2025.findings-naacl.30.pdf pages 1-2 (RAGate is a gating model for adaptive external knowledge augmentation). Turn 2 drops the subject and uses "it" and "the three variants", requiring the heuristic query rewriter to reconstruct "What are the three variants RAGate uses to implement the gating function?" before hitting the router. Pages 3-4 of 2025.findings-naacl.30.pdf (Section 3.2) name RAGate-Prompt, RAGate-PEFT, and RAGate-MHA explicitly. These three hyphenated names are unique to this paper. Re-using phase1 q5 verbatim confirms the rewriter still fires correctly under the router and that conversational context is preserved across turns.

---

### q6 — out_of_corpus

**Question:** What is the BLEU score achieved by HippoRAG 2 on the NarrativeQA benchmark when evaluated with a GPT-4o judge?
**Answer (excerpt, first 400 chars):** **Step 1: Analyze the Request** The user is asking for a specific quantitative metric: the BLEU score of HippoRAG 2 when tested on Narrative QA using GPT-4.  **Step 2: Scan the Provided Text for Keywords** I will search the text for the following keywords/phrases: *   "BLEU score" *   "HippoRAG 2" (or variations) *   "Narrative QA" *   "GPT-4"  **Step 3: Evaluate Findings** *   The text discusses 

**Sources returned:**
  - [1] 2502.14802v2 / 78.2 (retriever=router)
  - [2] 2502.14802v2 / Results (retriever=router)
  - [3] 2502.14802v2 / 74.7 (retriever=router)
  - [4] 2502.14802v2 / HippoRAG 2 (retriever=router)
  - [5] 2502.14802v2 / Discourse understanding evaluates sense-making by (retriever=router)
  - [6] 2502.14802v2 / 74.7 (retriever=router)

**Route distribution:** {'router': 10}
**Dominant route:** router
**KG path hit:** yes


**Missing substrings:** ["couldn't find"]

**Dimension scores:**
- substrings: FAIL
- citation: FAIL
- raft_blocks: FAIL
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: FAIL
- route_observed (info): router
- kg_path_hit (info): yes

**Score: FAIL**

**Notes:** None of the three papers report a BLEU score for HippoRAG 2 on NarrativeQA with a GPT-4o judge. 2502.14802v2.pdf does include NarrativeQA in its evaluation suite (Table 2, page 7) but reports F1 scores, not BLEU, and never mentions GPT-4o as a judge. The question is plausible — it combines real paper entities (HippoRAG 2, NarrativeQA) with a plausible-sounding metric — but the specific fact is absent. This avoids the phase1 q6 failure mode (citation FAIL because GPU passages were retrieved and cited): NarrativeQA F1 passages will be found and cited, but none will contain a "BLEU score ... GPT-4o judge" answer, forcing the "couldn't find" fallback on the substring check while the citation dimension can still pass cleanly.

---
