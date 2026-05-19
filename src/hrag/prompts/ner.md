## Named Entity Extraction

Extract all named entities, technical terms, and proper noun phrases from the
query below. Return a JSON list of strings — no prose, no markdown fences.

Lowercase + strip each entity. Prefer 1-4 word phrases. Do not include common
verbs, adjectives, or generic nouns.

### Examples

Query: "How does HippoRAG use Personalized PageRank to retrieve passages?"
Output: ["hipporag", "personalized pagerank"]

Query: "What is the architecture of BERT?"
Output: ["bert"]

### Your task

Query: {query}
Output:
