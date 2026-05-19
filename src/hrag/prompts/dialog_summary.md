## Dialog Summary

You are compacting a slice of an ongoing conversation so it can be carried
forward as context for the rest of the chat.

Below is a contiguous block of conversation turns between a user and an
assistant. Produce a single dense paragraph (no bullets, no markdown
headers) of approximately {summary_target_tokens} tokens that:

- Preserves named entities (people, projects, documents, code symbols,
  numeric values) verbatim.
- Records concrete decisions reached and unresolved questions still open.
- Uses neutral third-person voice — refer to "the user" and "the assistant".
- Stays faithful to the turns: do NOT invent facts, do NOT add citations,
  do NOT speculate about intent beyond what was said.

Examples of the target style:

Example A:
The user asked the assistant to compare PageRank and Personalized PageRank
on the project's knowledge graph. The assistant explained that PPR biases
the random walk toward a seed set so scores are query-dependent, and
recommended damping=0.85. The user agreed to use PPR and noted they still
need to decide how to pick seed entities — that question was left open.

Example B:
The user shared two PDFs about KG2RAG and HippoRAG and asked which the
project should follow first. The assistant suggested KG2RAG because the
codebase already builds a chunk MST, and proposed adding the redundancy
filter next. The user accepted and set the target for the Phase 2 sprint.

Now summarize the following turns in the same style.

Turns:
{turns}

Summary:
