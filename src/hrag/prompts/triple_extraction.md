## Triple Extraction Prompt (HippoRAG2-style OpenIE)

Extract all factual relationships from the passage as a JSON list of triples.

Each triple must be a JSON object with exactly three keys:
- `"head"`: the subject entity (noun phrase, 1-6 words)
- `"relation"`: the relationship verb or predicate (1-3 words)
- `"tail"`: the object entity (noun phrase, 1-6 words)

### Rules
- Prefer short, canonical noun phrases for head and tail (e.g. "Marie Curie",
  not "she" or "the famous scientist").
- Relation must be in active voice and as short as possible (e.g. "founded",
  "located in", "part of", "developed by").
- Extract every distinct factual relation; do not merge separate facts.
- Do not extract subjective opinions, conjunctions, or filler phrases.
- Output ONLY valid JSON — no prose, no markdown fences, no explanation.

### Examples

**Input passage:**
"Marie Curie was born in Warsaw and later moved to Paris, where she discovered
polonium and radium. She was the first woman to win a Nobel Prize."

**Output:**
[
  {"head": "Marie Curie", "relation": "born in", "tail": "Warsaw"},
  {"head": "Marie Curie", "relation": "moved to", "tail": "Paris"},
  {"head": "Marie Curie", "relation": "discovered", "tail": "polonium"},
  {"head": "Marie Curie", "relation": "discovered", "tail": "radium"},
  {"head": "Marie Curie", "relation": "first woman to win", "tail": "Nobel Prize"}
]

---

**Input passage:**
"The transformer architecture was introduced by Vaswani et al. in 2017 and
forms the basis of BERT and GPT models, both developed at large AI labs."

**Output:**
[
  {"head": "transformer architecture", "relation": "introduced by", "tail": "Vaswani et al."},
  {"head": "transformer architecture", "relation": "introduced in", "tail": "2017"},
  {"head": "BERT", "relation": "based on", "tail": "transformer architecture"},
  {"head": "GPT", "relation": "based on", "tail": "transformer architecture"},
  {"head": "BERT", "relation": "developed at", "tail": "large AI labs"},
  {"head": "GPT", "relation": "developed at", "tail": "large AI labs"}
]

---

### Your task

Passage:
{passage}

Output:
