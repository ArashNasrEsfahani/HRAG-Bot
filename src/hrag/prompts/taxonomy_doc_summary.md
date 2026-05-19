## Taxonomy Doc Summary

You are writing a single-sentence TOPIC summary of a document for a hierarchical
category tree. The summary will be embedded and clustered, so it must describe
*what the document is about* — not its author, style, or formatting.

### Rules
- Output ONE LINE of plain text. No quotes, no preamble, no "Summary:" prefix,
  no markdown.
- Maximum 280 characters.
- Focus on TOPIC and SUBJECT MATTER. If the document is a research paper, name
  the main subject and its core contribution (e.g. "A method for X using Y").
- Use concrete nouns, not vague abstractions ("graph neural networks" not
  "advanced techniques").
- Do not mention the author, venue, year, or document type unless that *is* the
  topic.

### Examples

**Title:** Attention Is All You Need
**Excerpt:** The dominant sequence transduction models are based on complex
recurrent or convolutional neural networks... We propose a new simple network
architecture, the Transformer, based solely on attention mechanisms...
**Output:** The Transformer architecture for sequence transduction using
self-attention in place of recurrence and convolution.

---

**Title:** Soft Actor-Critic for Robotic Manipulation
**Excerpt:** We apply maximum-entropy reinforcement learning to dexterous
manipulation tasks on a 7-DoF arm...
**Output:** Soft actor-critic reinforcement learning applied to dexterous
robotic manipulation with a 7-DoF arm.

---

### Your task

**Title:** {title}

**Excerpt:**
{excerpt}

**Output:**
