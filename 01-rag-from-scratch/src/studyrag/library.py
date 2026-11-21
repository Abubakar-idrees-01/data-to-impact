"""The document library: a persistent, incrementally-updated index on disk.

storage/
  manifest.json    which documents are indexed (+ file hashes) and with which settings
  chunks.jsonl     one chunk per line (text + location metadata)
  embeddings.npy   one row per chunk, same order as chunks.jsonl

Incremental indexing: adding a document embeds only *its* chunks; removing one drops its rows.
Embedding is the expensive step, so this is what keeps "add a file" fast. BM25 is rebuilt in
memory on load because it takes milliseconds. If you change the chunking strategy or embedding
model, the manifest no longer matches and the library re-indexes from the original files.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25Index
from .chunking import Chunk, chunk_document
from .config import Settings
from .ingest import SUPPORTED, file_hash, load_document
from .text import tokenize
from .vector_index import VectorIndex

INDEX_VERSION = 2       # bump when loaders change what text a file produces (2: linearized tables)


@dataclass
class DocInfo:
    doc_id: str
    title: str
    source: str
    path: str
    doc_type: str
    n_chunks: int
    n_words: int
    added_at: float
    n_ocr: int = 0                       # slides/pages whose text came from OCR
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Library:
    def __init__(self, settings: Settings, embedder):
        self.settings = settings
        self.embedder = embedder
        self.dir = Path(settings.storage_dir)
        self.docs: dict[str, DocInfo] = {}
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, embedder.dim), dtype=np.float32)
        self.bm25 = BM25Index()
        self.vindex = VectorIndex()
        from .ingest import ocr as ocr_mod
        ocr_mod.set_cache_path(Path(settings.ocr_cache_file))
        self._load()

    # ------------------------------------------------------------ persistence
    def _signature(self) -> dict:
        s = self.settings
        return {"version": INDEX_VERSION, "embedding_model": self.embedder.model_name,
                "chunk_strategy": s.chunk_strategy, "chunk_size": s.chunk_size,
                "chunk_overlap": s.chunk_overlap, "min_chunk_words": s.min_chunk_words, "ocr": s.ocr}

    def _load(self) -> None:
        manifest = self.dir / "manifest.json"
        if not manifest.exists():
            return
        data = json.loads(manifest.read_text())
        docs = {k: DocInfo(**v) for k, v in data["documents"].items()}
        if data.get("signature") != self._signature():
            # settings changed: re-index every document from its original file
            paths = [d.path for d in docs.values() if Path(d.path).exists()]
            for p in paths:
                self.add_file(p, save=False)
            self._save()
            return
        self.docs = docs
        with open(self.dir / "chunks.jsonl") as f:
            self.chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        self.matrix = np.load(self.dir / "embeddings.npy")
        self._rebuild_indexes()

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        manifest = {"signature": self._signature(), "documents": {k: d.to_dict() for k, d in self.docs.items()}}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with open(self.dir / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(self.dir / "embeddings.npy", self.matrix)

    def _rebuild_indexes(self) -> None:
        self.bm25 = BM25Index().build([tokenize(c.index_text) for c in self.chunks])
        self.vindex = VectorIndex(self.matrix) if len(self.chunks) else VectorIndex()

    # ------------------------------------------------------------ mutations
    def add_file(self, path: str | Path, save: bool = True) -> tuple[str, str]:
        """Index one file. Returns (doc_id, status) with status 'added' | 'updated' | 'unchanged'."""
        path = Path(path).resolve()
        h = file_hash(path)
        if h in self.docs:
            return h, "unchanged"
        status = "added"
        for d in list(self.docs.values()):            # same file path, new content -> replace
            if Path(d.path) == path:
                self.remove(d.doc_id, save=False)
                status = "updated"
        s = self.settings
        doc = load_document(path, ocr=s.ocr)
        chunks = chunk_document(doc, s.chunk_strategy, size=s.chunk_size, overlap=s.chunk_overlap,
                                min_words=s.min_chunk_words, embed=self.embedder.embed_documents)
        vecs = self.embedder.embed_documents([c.index_text for c in chunks])
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, vecs]) if len(self.matrix) else vecs
        self.docs[doc.doc_id] = DocInfo(doc_id=doc.doc_id, title=doc.title, source=doc.source,
                                        path=str(path), doc_type=doc.doc_type, n_chunks=len(chunks),
                                        n_words=doc.word_count, added_at=time.time(),
                                        n_ocr=doc.ocr_blocks, warnings=doc.warnings)
        self._rebuild_indexes()
        if save:
            self._save()
        return doc.doc_id, status

    def remove(self, doc_id: str, save: bool = True) -> None:
        if doc_id not in self.docs:
            return
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.matrix = self.matrix[keep] if keep else np.zeros((0, self.embedder.dim), dtype=np.float32)
        del self.docs[doc_id]
        self._rebuild_indexes()
        if save:
            self._save()

    def sync_folder(self, folder: str | Path, remove_missing: bool = True) -> dict[str, list[str]]:
        """Make the library mirror a folder: index new/changed files, drop deleted ones."""
        folder = Path(folder).resolve()
        report: dict[str, list[str]] = {"added": [], "updated": [], "unchanged": [], "removed": [], "failed": []}
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED and p.is_file()
                       and not p.name.startswith((".", "~$")) and p.name.lower() != "readme.md")
        for p in files:
            try:
                _, status = self.add_file(p, save=False)
                report[status].append(p.name)
            except Exception as e:  # one bad file shouldn't stop the rest
                report["failed"].append(f"{p.name}: {e}")
        if remove_missing:
            present = {str(p.resolve()) for p in files}
            for d in list(self.docs.values()):
                if d.path.startswith(str(folder)) and d.path not in present:
                    self.remove(d.doc_id, save=False)
                    report["removed"].append(d.source)
        self._save()
        return report

    # ------------------------------------------------------------ queries
    def allowed_mask(self, doc_ids: list[str] | None) -> np.ndarray | None:
        if not doc_ids:
            return None
        wanted = set(doc_ids)
        return np.array([c.doc_id in wanted for c in self.chunks], dtype=bool)

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def find_doc(self, name_or_id: str) -> DocInfo | None:
        if name_or_id in self.docs:
            return self.docs[name_or_id]
        q = name_or_id.lower()
        for d in self.docs.values():
            if q in d.title.lower() or q == d.source.lower():
                return d
        return None
