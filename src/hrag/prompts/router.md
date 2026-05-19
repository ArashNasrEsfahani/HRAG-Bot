## Query Routing Classifier

Classify the user's question into ONE of these categories:

- entity      : The question asks about a specific named thing, person, or
                technique. Best answered by traversing entity links in the KG.
                Example: "What is the PPR damping factor in HippoRAG?"

- global      : The question asks for a high-level summary, theme, or pattern
                across the corpus. Best answered by community summaries.
                Example: "What are the main themes across these papers?"

- cross_document : The question explicitly compares or relates entities from
                   different documents. Needs both entity hops AND breadth.
                   Example: "How do HippoRAG and RAGate differ in approach?"

- ambiguous   : The question is short, vague, or could be answered multiple
                ways. Run multiple retrievers and merge.
                Example: "Tell me more about retrieval."

When uncertain or when the question names a specific technique, dataset, paper, hyperparameter, or method, prefer: entity.

Output exactly ONE of these labels: entity | global | cross_document | ambiguous

### Examples

Q: "What dataset does RAGate use for evaluation?"
A: entity

Q: "Summarize the key contributions of all three papers."
A: global

Q: "How does HippoRAG's PPR algorithm compare to RAGate's gating mechanism?"
A: cross_document

Q: "Tell me more about that."
A: ambiguous

Q: "What is the synonym threshold in HippoRAG?"
A: entity

Q: "What dataset does RAGate use for training?"
A: entity

Q: "What three components does HippoRAG model its architecture on?"
A: entity

### Your task

Q: {query}
A:
