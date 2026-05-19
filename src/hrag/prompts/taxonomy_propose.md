## Taxonomy Proposal

You are organising a user's document collection into a hierarchical taxonomy
tree of THEMATIC categories. Each leaf node holds a set of documents that share
a topic.

### Hard rules
- Branching factor: at most {max_children} children per node.
- Maximum depth: {max_depth} (root counts as depth 0; deepest leaf depth must
  not exceed this).
- Categories must reflect TOPIC / SUBJECT MATTER (e.g. "Reinforcement Learning"
  > "Robotics" > "Manipulation"), not file metadata (date, author, filetype).
- Every input `doc_id` MUST appear under exactly one leaf, in the leaf's
  `doc_ids` array.
- Empty leaves are forbidden. Every leaf must contain at least one `doc_id`.
- Internal nodes (with children) must NOT carry `doc_ids` — only leaves do.
- Output JSON ONLY. No prose, no markdown fences, no commentary.

### Output schema
```
{{
  "tree": {{
    "label": "root",
    "children": [
      {{
        "label": "Category A",
        "description": "one-line description of the category",
        "children": [
          {{
            "label": "Subcategory A1",
            "description": "one-line description",
            "doc_ids": ["doc_id_1", "doc_id_2"]
          }}
        ]
      }}
    ]
  }}
}}
```

Leaf nodes use the key `doc_ids` and omit `children`. Internal nodes use
`children` and omit `doc_ids`. The root must be labelled exactly `"root"`.

### Input documents
Each line is tab-separated: `<doc_id>\t<title>\t<summary>`.

{doc_list}

### Your output
Return the JSON tree only.
