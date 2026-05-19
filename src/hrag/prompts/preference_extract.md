## Preference Extraction Prompt (Phase 3 User Profile)

Extract user preferences and personal facts from the conversation below.

Output a JSON list. Each item must have exactly these four keys:

```
{
  "polarity": "like" | "dislike" | "fact" | "style",
  "topic":    "<topic or domain, short noun phrase>",
  "value":    "<what the user said or prefers, concise>",
  "confidence": <float 0.0–1.0>
}
```

### Polarity definitions
- `"like"` — user explicitly expressed positive preference about something
  ("I love...", "I prefer...", "I'd rather have...")
- `"dislike"` — user explicitly expressed negative preference
  ("I hate...", "please avoid...", "I don't like...")
- `"fact"` — user volunteered a personal fact about themselves
  (job, location, language, background, constraints)
- `"style"` — user asked for a different response style
  ("keep it short", "be more technical", "use bullet points")

### Rules
- Only extract things the **user** said about **themselves**. Skip anything
  that is purely discussion of document content.
- Confidence should reflect how explicit the statement was:
  - 0.9-1.0: direct and unambiguous statement
  - 0.6-0.8: implied or hedged ("I think I prefer...", "usually I like...")
  - 0.3-0.5: inferred from behaviour or indirect phrasing
- If nothing extractable is present, return an empty list: `[]`
- Output ONLY valid JSON. No prose, no markdown fences, no explanation.

### Example

**Conversation:**
"User: Can you keep the answers shorter? I'm a data engineer so I don't need
basic SQL explained. I usually prefer Python examples over R.
Assistant: Got it, I'll be concise and use Python.
User: Also, I'm based in Singapore so use SGD for any cost examples."

**Output:**
[
  {"polarity": "style",   "topic": "response length",   "value": "shorter answers",           "confidence": 1.0},
  {"polarity": "fact",    "topic": "occupation",         "value": "data engineer",             "confidence": 1.0},
  {"polarity": "dislike", "topic": "basic SQL explanation", "value": "does not need basics",   "confidence": 0.9},
  {"polarity": "like",    "topic": "code language",      "value": "Python over R",             "confidence": 0.8},
  {"polarity": "fact",    "topic": "location",           "value": "Singapore",                 "confidence": 1.0},
  {"polarity": "style",   "topic": "currency in examples", "value": "SGD",                    "confidence": 1.0}
]

---

### Your task

Conversation:
{conversation}

Output:
