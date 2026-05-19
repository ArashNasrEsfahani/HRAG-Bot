## Clue Prompt (MemoRAG-style Retrieval Hypothesis)

Your task is to write a short retrieval hypothesis — a draft outline of what a
good answer to the question would look like — so it can be used as a semantic
search query against a document store.

**Do NOT write the actual answer.** Write what the answer probably looks like:
which entities, concepts, figures, or relationships it would mention, and what
form the information would take (e.g. a table of numbers, a step-by-step
process, a definition, a list of names).

Output 2-4 sentences. Be specific about the likely entities and topics involved.
Use noun phrases and domain vocabulary the source documents would contain.

---

### Examples

Conversation: ""
Question: "What safety protocols apply when handling lithium batteries?"
Hypothesis: A complete answer would enumerate specific handling and storage
protocols for lithium-ion or lithium-polymer cells, likely referencing
temperature limits, short-circuit prevention measures, and approved disposal
methods. It would probably cite a safety data sheet (SDS) or internal policy
document and name relevant regulatory standards such as IEC 62133 or UN 38.3.

---

Conversation: "User: I'm looking into our Q2 performance.\nAssistant: Sure, I
can help with that."
Question: "How did the APAC region do compared to EMEA?"
Hypothesis: The answer would present revenue or growth figures for the APAC and
EMEA regions for Q2, likely drawn from an earnings report or regional
performance dashboard. It would reference specific metrics such as YoY growth
percentage, headcount, or ARR, and may include a comparison table or breakdown
by sub-region.

---

### Your task

Conversation:
{conversation}

Question: {question}

Hypothesis:
