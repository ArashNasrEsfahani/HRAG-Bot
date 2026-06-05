You are deep-reading the document "{doc_title}" to answer the user's question,
one navigation step at a time. Use ONLY what you read — no outside knowledge.

### Question
{question}

### Document map (each line is a part you can open by its number)
{structure}

### What you've learned so far
{notes_so_far}

### Passages you just read
{passages}

---

Choose ONE next action and reply with ONLY a JSON object:

- Open a part you have NOT read yet that likely holds the answer:
  {{"note": "what the last passages added", "action": "read_part", "part_idx": 6}}
- Search this document for a specific phrase when no single part obviously fits:
  {{"note": "...", "action": "search", "query": "confrontation with the unconscious"}}
- Stop and write the answer once the question is well covered:
  {{"note": "...", "action": "answer"}}

Rules:
- `note` is 1–2 sentences grounded in the passages above (use "" if they added nothing).
- Prefer `read_part` when the map shows an unread part likely to contain the answer —
  that is how you move on to another chapter/section of the document.
- Do NOT re-open a part already marked READ.
- Reply with exactly ONE JSON object and nothing else.
