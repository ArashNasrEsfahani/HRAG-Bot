# Phase 1 Benchmark Results

Run timestamp: 2026-05-06T10:46:53+00:00
Total wall time: 410.3s
Pass rate: 4/6 questions

## Summary table

| id | category | overall | substrings | citation | raft | src match | no fallback | rewriter | no halluc | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| q1 | factual_single_source | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 44.6 |
| q2 | factual_single_source | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 43.6 |
| q3 | multi_hop | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 80.9 |
| q4 | cross_document | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | PASS | 72.1 |
| q5 | conversational_followup | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | 117.4 |
| q6 | out_of_corpus | FAIL | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | 51.6 |

## Per-question detail

### q1 — factual_single_source

**Question:** What three components of human long-term memory does HippoRAG model its architecture on?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for the three components of human long-term memory that HippoRAG models its architecture on. Source 1 provides a direct and explicit list of these three primary components. It states that the framework consists of: 1) an LLM acting as an artificial neocortex; 2) a KG and the Personalized PageRank algorithm mirroring the hippocampus; and 3) a retrieval encoder reflectin

**Sources returned:**
  - [1] 2502.14802v2 / HippoRAG 2
  - [2] HIPPORAG / NER
  - [3] HIPPORAG / 2.2
  - [4] HIPPORAG / 2.1
  - [5] HIPPORAG / N/A
  - [6] HIPPORAG / 6.3


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS

**Score: PASS**

**Notes:** Pages 2-3 of HIPPORAG.pdf describe the hippocampal memory indexing theory: neocortex processes stimuli, parahippocampal regions route them, and the hippocampus indexes them. These three proper nouns are verbatim in the paper and unlikely to appear by coincidence. Tests basic single-source factual retrieval.

---

### q2 — factual_single_source

**Question:** What dataset does RAGate use for training and evaluation, and how many dialogues does it contain?
**Answer (excerpt, first 400 chars):** Reasoning: The user asks for the dataset used for RAGate's training and evaluation, and the number of dialogues it contains.  1.  **Identify Dataset Name and Dialogue Count:** Source 1 explicitly states the dataset used for evaluation and provides the required metrics. It says, "Specifically, we use the KE-TOD dataset (Chen et al., 2022), which has fully annotated 5,324 dialogues and 52,063 turns 

**Sources returned:**
  - [1] 2025.findings-naacl.30 / (2)
  - [2] 2025.findings-naacl.30 / 5.1
  - [3] 2025.findings-naacl.30 / 9.3425
  - [4] 2025.findings-naacl.30 / 5.3
  - [5] 2025.findings-naacl.30 / 9.3425
  - [6] 2502.14802v2 / 4, 111


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS

**Score: PASS**

**Notes:** Pages 4-5 of 2025.findings-naacl.30.pdf: 'we use the KETOD dataset (Chen et al., 2022), which has fully annotated 5,324 dialogues'. KETOD is a distinctive dataset name and 5,324 is a specific number — both are robust identifiers that won't appear in the other two papers.

---

### q3 — multi_hop

**Question:** In HippoRAG, what algorithm is used during online retrieval, and what two hyperparameters were tuned using MuSiQue training data?
**Answer (excerpt, first 400 chars):** Reasoning: The user asks for two pieces of information: the algorithm used during online retrieval in HippoRAG, and the two hyperparameters tuned using MuSiQue training data.  1.  **Identifying the Algorithm:** Source 8 explains the mechanism of HippoRAG's retrieval process. It states that "HippoRAG identifies the key concepts in the query and runs the Personalized PageRank (PPR) algorithm [30] on

**Sources returned:**
  - [1] HIPPORAG / 3.4
  - [2] HIPPORAG / 51.7
  - [3] HIPPORAG / 74.6
  - [4] HIPPORAG / 3.4
  - [5] HIPPORAG / 72.9
  - [6] HIPPORAG / 3.3


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS

**Score: PASS**

**Notes:** Requires combining two chunks from HIPPORAG.pdf: the overview (pages 3-4) names Personalized PageRank as the retrieval algorithm, while the implementation details (page 6, section 3.4) states the two hyperparameters tuned are the synonymy threshold (0.8) and the PPR damping factor (0.5). Neither chunk alone answers the full question.

---

### q4 — cross_document

**Question:** Both HippoRAG and RAGate address a limitation of standard RAG. What limitation does HippoRAG address, and what limitation does RAGate address?
**Answer (excerpt, first 400 chars):** Reasoning: The question asks for the specific limitations addressed by HippoRAG and RAGate, respectively.  For HippoRAG, [Source 2] provides a direct definition of a limitation that standard RAG cannot handle: "Associativity is the capacity to draw multi-hop connections between disparate facts for efficient retrieval. It is an important part of continual learning, which standard RAG cannot emulate

**Sources returned:**
  - [1] 2502.14802v2 / Introduction
  - [2] 2502.14802v2 / Related Work
  - [3] 2502.14802v2 / Discussions
  - [4] 2502.14802v2 / 44.3 / 58.1
  - [5] HIPPORAG / 52.0
  - [6] 2502.14802v2 / Hull County, Quebec


**Missing substrings:** ['knowledge integration', 'every turn']

**Dimension scores:**
- substrings: FAIL
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS

**Score: FAIL**

**Notes:** Requires synthesizing from two papers: HIPPORAG.pdf (pages 1-2) states HippoRAG addresses the inability of standard RAG to efficiently integrate new experiences (knowledge integration / multi-hop). 2025.findings-naacl.30.pdf (pages 1-2) states RAGate addresses the unnecessary cost of augmenting every turn of the conversation. 'knowledge integration' is HippoRAG's key phrase; 'every turn' is RAGate's key phrase. Both must appear for a correct cross-document answer.

---

### q5 — conversational_followup

**Turn 1:** What is RAGate and what problem does it solve?
**Turn 2:** What are the three variants it uses to implement the gating function?
**Answer (excerpt, first 400 chars):** [T1] Reasoning: The user is asking for a definition of RAGate and the problem it solves.  1.  **Definition of RAGate:** Source 5 provides the most direct definition. It states that RAGate is a "gating model" developed by leveraging human judgments on the binary choice of adaptive augmentation. Its function is to "model conversation context and relevant inputs to predict if a conversational system 

**Sources returned:**
  - [1] 2025.findings-naacl.30 / 5.1
  - [2] 2025.findings-naacl.30 / 5.3
  - [3] 2025.findings-naacl.30 / 5.2
  - [4] 2025.findings-naacl.30 / EMNLP.
  - [5] 2025.findings-naacl.30 / 9.3425
  - [6] 2025.findings-naacl.30 / 0.0915

**Query rewrite:** `What are the three variants it uses to implement the gating function?` → `What is RAGate and what problem does it solve?
What are the three variants it uses to implement the gating function?`

**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- citation: PASS
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS

**Score: PASS**

**Notes:** Turn 1 is a factual question about RAGate from pages 1-2 of 2025.findings-naacl.30.pdf. Turn 2 uses 'it' and 'the three variants' without restating 'RAGate' — the heuristic query rewriter must expand it to 'What are the three variants RAGate uses to implement the gating function?' Pages 3-4 name RAGate-Prompt, RAGate-PEFT, and RAGate-MHA. These three hyphenated names are unique and specific; none appear in the HippoRAG papers.

---

### q6 — out_of_corpus

**Question:** What is the GPU memory footprint in gigabytes of HippoRAG when running on a 7B Llama model?
**Answer (excerpt, first 400 chars):** Reasoning: None of the retrieved passages provide the specific GPU memory footprint in gigabytes for HippoRAG when running on a 7B Llama model. While several passages discuss Llama models and compute requirements, Source 6 mentions using 4 NVIDIA H100 GPUs with 80GB of memory for indexing with Llama-3.1 models, but this does not specify the memory footprint for a 7B model, nor does it link that me

**Sources returned:**
  - [1] 2502.14802v2 / 78.2
  - [2] HIPPORAG / 5.1
  - [3] HIPPORAG / 20-40
  - [4] 2502.14802v2 / 44.3 / 58.1
  - [5] 2025.findings-naacl.30 / NVIDIA 4090 GPU.
  - [6] HIPPORAG / 20-40


**Missing substrings:** (none)

**Dimension scores:**
- substrings: PASS
- citation: FAIL
- raft_blocks: PASS
- source_doc_match: PASS
- rerank_fallback_avoided: PASS
- rewriter_fired_on_followup: PASS
- no_hallucination: PASS

**Score: FAIL**

**Notes:** None of the three papers (HIPPORAG.pdf, 2502.14802v2.pdf, 2025.findings-naacl.30.pdf) report GPU memory footprint figures for HippoRAG. The question is plausible-sounding but the answer is genuinely absent from the corpus. Tests that the system falls back to the 'I couldn't find that in your documents' path rather than hallucinating a number.

---
