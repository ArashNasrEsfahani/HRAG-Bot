"""Phase 3 per-user memory layer.

Two stores:
- EpisodicMemoryStore (memory.store): one note = one document with source_type='episodic'.
  Reuses IngestPipeline so memories share the chunks/Chroma index and compete with
  documents in retrieval by default.
- ProfileStore (memory.profile): CRUD over the structured `preferences` table.
  Rendered verbatim into the answer prompt via the {user_profile} placeholder.
"""

from hrag.memory.extractor import PreferenceCandidate, PreferenceExtractor
from hrag.memory.profile import Preference, ProfileStore
from hrag.memory.store import EpisodicMemoryStore

__all__ = [
    "EpisodicMemoryStore",
    "Preference",
    "PreferenceCandidate",
    "PreferenceExtractor",
    "ProfileStore",
]
