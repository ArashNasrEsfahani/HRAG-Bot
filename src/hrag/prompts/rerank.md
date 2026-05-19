## Rerank Prompt (ACP-RAG 0-3 Relevance Scorer)

Score how relevant the passage is to answering the query.

Output ONLY a single integer: 0, 1, 2, or 3. No explanation. No punctuation.

### Scoring rubric

| Score | Meaning |
|-------|---------|
| 0 | Irrelevant — unrelated topic, wrong domain, or pure noise |
| 1 | Tangentially related — shares vocabulary or domain but contains no useful information for this query |
| 2 | Related — provides useful context or background that partially supports an answer |
| 3 | Directly answers — contains the specific fact, definition, or data the query is asking for |

### Examples

Query: "What is the boiling point of ethanol?"
Passage: "Ethanol is widely used as a solvent in pharmaceutical manufacturing."
Score: 1

Query: "What is the boiling point of ethanol?"
Passage: "Common solvents include acetone, methanol, and ethanol, each with distinct physical properties used in laboratory settings."
Score: 2

Query: "What is the boiling point of ethanol?"
Passage: "Ethanol (C2H5OH) has a boiling point of 78.37 °C at standard atmospheric pressure."
Score: 3

Query: "What is the boiling point of ethanol?"
Passage: "The annual report shows a 12% increase in revenue driven by digital product sales."
Score: 0

---

### Your task

Query: {query}

Passage: {passage}

Score:
