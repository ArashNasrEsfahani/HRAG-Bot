## Intent Classification

Classify the user's message into ONE of four intents. Output ONLY the lowercase intent word, with no other text, no quotes, no punctuation.

### Intents

- `greeting` — pleasantries and short acknowledgements: "hi", "hey", "thanks", "good morning", "bye", "ok cool".
- `personal` — questions about **the user themselves OR about people / topics already established as personal in the conversation so far** (their name, identity, what we know or remember about *them*, or follow-up questions about a friend / contact the user mentioned earlier). Examples: "what's my name", "who am I", "do you remember me", "tell me about myself", "what do you know about me". Follow-up example: if the prior assistant turn answered with "Mahmoud is your friend", then a next user message like "what does Mahmoud do?" is `personal` — it is a continuation of the same personal thread.
- `factual` — questions about **topics, papers, concepts, or named entities** that live in documents (NOT about the user). Examples: "what is RAG", "what do you know about HippoRAG", "explain pagerank", "summarize the paper on memory networks", "tell me about transformers", "search the documents for HippoRAG".
- `unclear` — gibberish, empty, single punctuation, or genuinely ambiguous.

### CRITICAL DISTINCTION — `personal` vs `factual`

Only classify as `personal` when the message refers to the **user themselves** with words like "me", "I", "my", "myself", **OR** when the message is a follow-up about a person or topic the conversation history already framed as personal (someone the user mentioned, a friend, a colleague, a relationship).

- "what do you know about **me**" → `personal`   (the word "me" anchors it on the user)
- "what do you know about **HippoRAG**" → `factual` (HippoRAG is a topic/entity)
- "what do you know about **transformers**" → `factual`
- "tell me about **myself**" → `personal`
- "tell me about **transformers**" → `factual`
- "search the documents for **it**" → `factual` (the user is asking about a topic, not themselves)
- "search **my** documents" → `factual` (asking about docs, not about identity)
- Prior turn: user "who am I?" → assistant "You are Arash, and Mahmoud is your friend." Next user message "what about Mahmoud?" → `personal` (the conversation context anchors it to a person from the user's life).

If the message is about a **named technology, paper, person who is not the user, or general topic** AND the conversation history does NOT frame that person as someone the user knows, it is `factual` — even if the message uses phrasing like "what do you know about X".

### Examples

- `"heeey"` → greeting
- `"yo what's up"` → greeting
- `"thanks"` → greeting
- `"tell me about myself"` → personal
- `"what do you know about me"` → personal
- `"do you remember my name"` → personal
- `"what do you know about HippoRAG"` → factual
- `"what is RAG"` → factual
- `"explain personalized pagerank"` → factual
- `"summarize the paper on attention"` → factual
- `"search the documents for it"` → factual
- `"tell me about the NAACL paper"` → factual
- `"asdfgh"` → unclear
- `""` → unclear

### Recent conversation (oldest first)
{conversation_history}

User message: {query}

Output (one word only):
