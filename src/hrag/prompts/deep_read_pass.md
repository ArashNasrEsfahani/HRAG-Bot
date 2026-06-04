You are deep-reading the document "{doc_title}" to answer the user's question,
one pass at a time. Use ONLY the passages below — no outside knowledge.

### Question
{question}

### What you've learned so far
{notes_so_far}

### New passages just read in this pass
{passages}

---

Reply with ONLY a JSON object with three fields:

1. `note` — 1–3 sentences on what THESE new passages add toward answering the
   question. Ground every claim in the passages. If they add little, say so.
2. `next_query` — the single most useful thing to look for NEXT *in this
   document* to deepen the answer: a short search phrase based on a gap you
   noticed (e.g. "confrontation with the unconscious"). If the question is now
   well covered, use an empty string "".
3. `done` — `true` if the document is covered well enough to answer, OR this
   pass added little; otherwise `false`.

Reply with exactly this shape, nothing else:
{{"note": "...", "next_query": "...", "done": false}}
