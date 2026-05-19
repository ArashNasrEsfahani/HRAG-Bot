## Taxonomy Route Tiebreak

You are routing a query (or document summary) down a category tree. Two or more
child categories scored very closely by cosine similarity, so you must pick the
single best match.

### Rules
- Output ONLY the integer index (0-based) of the best-matching candidate.
- No prose, no JSON, no quotes, no preamble — just one integer.
- If genuinely ambiguous, prefer the candidate whose label/description most
  specifically covers the query's topic.

### Query
{query}

### Candidates
{candidates}

### Your output (a single integer)
