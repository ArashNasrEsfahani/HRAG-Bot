## Batched Rerank Prompt (ACP-RAG 0-3 Relevance Scorer)

Score how relevant EACH passage is to answering the query.

Output ONLY a JSON array of integers — one integer per passage, in the same order.
No prose. No markdown fences. No keys. Just the array, e.g. `[2, 0, 3]`.
Length of the array MUST equal the number of passages. If unsure, return 0.

### Scoring rubric

| Score | Meaning |
|-------|---------|
| 0 | Irrelevant — unrelated topic, wrong domain, or pure noise |
| 1 | Tangentially related — shares vocabulary but contains no useful information |
| 2 | Related — provides useful context that partially supports an answer |
| 3 | Directly answers — contains the specific fact or data the query is asking for |

### Example

Query: "What is the boiling point of ethanol?"

[1] Ethanol is widely used as a solvent in pharmaceutical manufacturing.

[2] The annual report shows a 12% increase in revenue driven by digital product sales.

[3] Ethanol (C2H5OH) has a boiling point of 78.37 °C at standard atmospheric pressure.

Output: [1, 0, 3]

---

### Your task

Query: {query}

{numbered_passages}

Output:
