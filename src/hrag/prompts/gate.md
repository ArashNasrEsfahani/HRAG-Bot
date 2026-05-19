## Gate Prompt (RAGate-Prompt Retrieval Classifier)

Decide whether the current question requires retrieving documents.

Output ONLY one word: `RETRIEVE` or `SKIP`. No explanation. No punctuation.

### Rules

Output `RETRIEVE` when the question:
- Asks for a specific fact, figure, date, name, or definition
- References a document, report, policy, topic, or named entity
- Requires up-to-date or domain-specific information not available from the
  conversation alone
- Is ambiguous but could plausibly need a document lookup

Output `SKIP` when the question:
- Is a greeting, farewell, or small talk ("Hi", "Thanks", "How are you?")
- Asks about the bot itself ("What can you do?", "Are you an AI?")
- Is fully answerable from the conversation history with no need for documents
- Is a simple clarification or acknowledgement with no new information need

When in doubt, prefer `RETRIEVE`.

---

### Examples

Conversation: "User: Hello!\nAssistant: Hi! How can I help?"
Question: "What's the refund policy for enterprise plans?"
Output: RETRIEVE

Conversation: "User: What is the capital of France?\nAssistant: Paris."
Question: "Thanks, that's all I needed!"
Output: SKIP

Conversation: ""
Question: "Summarise the Q3 earnings report."
Output: RETRIEVE

Conversation: "User: Hi there.\nAssistant: Hello! I'm your document assistant."
Question: "Can you speak French?"
Output: SKIP

---

### Your task

Conversation:
{conversation}

Question: {question}

Output:
