"""
services/rag_retrieval_service/hybrid_engine.py
Implementiert den vollständigen Hybrid-Retriever gemäß HOMIE-Architektur:
- Dense Retrieval: ChromaDB (HNSW) + BAAI/bge-m3 (CPU)
- Lexical Retrieval: BM25Okapi
- Fusion: Reciprocal Rank Fusion (RRF)
- Reranking: Cross-Encoder (BAAI/bge-reranker-v2-m3) auf sauberem Text (Titel + text_raw) + Dense-Kalibrierung
"""

import json
import math
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from shared.logger import get_service_logger
from shared.schemas import (
    DocumentChunk,
    LanguageLevel,
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResponse,
    ScoredChunk,
    SourceStatus,
)

logger = get_service_logger("rag-hybrid-engine")

BUNDLE_DIR = Path(os.getenv("PRODUCTION_BUNDLE_DIR", "data/production_bundle"))
CHROMA_DIR = BUNDLE_DIR / "chroma_db"
BM25_FILE = BUNDLE_DIR / "bm25_store.json"
# Früheres Format (pickle). Wird nicht mehr geladen: pickle.load kann beim Laden beliebigen Code ausführen,
# und das Bündel wird per USB/OTA auf den Roboter verteilt (§ 11 C b). Nach dem Update einmal neu synchronisieren.
LEGACY_BM25_PICKLE = BUNDLE_DIR / "bm25_store.pkl"
MODELS_CACHE_DIR = Path(os.getenv("MODELS_CACHE_DIR", "data/models_cache"))

for directory in (BUNDLE_DIR, CHROMA_DIR, MODELS_CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")


class HybridRetrievalEngine:
    def __init__(self):
        logger.info("Initialisiere HybridRetrievalEngine (CPU-Modus zur VRAM-Schonung)...")

        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="homie_quartier_wissen",
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"Lade Embedding-Modell '{EMBEDDING_MODEL_NAME}' auf CPU...")
        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu",
            cache_folder=str(MODELS_CACHE_DIR),
        )

        logger.info(f"Lade Cross-Encoder Reranker '{RERANKER_MODEL_NAME}' auf CPU...")
        self.reranker, self._reranker_returns_logits = self._load_reranker(RERANKER_MODEL_NAME)

        self.chunk_store: Dict[str, DocumentChunk] = {}
        self.bm25_ids: List[str] = []
        self.bm25_index: Optional[BM25Okapi] = None
        self._load_bm25_store()

        logger.info(
            f"HybridRetrievalEngine bereit. Aktuell indexierte Chunks: {len(self.chunk_store)}"
        )

    @staticmethod
    def _load_reranker(model_name: str) -> Tuple[CrossEncoder, bool]:
        """
        Lädt den Cross-Encoder so, dass predict() rohe Logits liefert.
        Hintergrund: sentence-transformers wendet bei Modellen mit einem Label standardmäßig schon
        eine Sigmoid-Funktion an. Die zweite Sigmoid-Funktion in retrieve() hat jeden Wert in den
        Bereich 0,5 bis 0,73 geschoben; mit 0,7 * rerank + 0,3 * dense lag der Score damit immer
        über 0,35, und der Quellen-Guard (Schwelle 0,25) hat nie ausgelöst.
        Rückgabe: (Modell, True wenn predict() Logits liefert).
        """
        import torch

        for kwarg in ("activation_fn", "default_activation_function"):  # Name je nach Bibliotheksversion
            try:
                model = CrossEncoder(
                    model_name, device="cpu", max_length=512, **{kwarg: torch.nn.Identity()}
                )
                return model, True
            except TypeError:
                continue
        logger.warning("Cross-Encoder ohne Identity-Aktivierung geladen; Scores gelten als Wahrscheinlichkeiten.")
        return CrossEncoder(model_name, device="cpu", max_length=512), False

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\säöüßÄÖÜçğıöşüÇĞİÖŞÜ]", " ", text.lower())
        return [tok for tok in cleaned.split() if len(tok) > 1]

    def _rebuild_bm25_in_memory(self) -> None:
        self.bm25_ids = list(self.chunk_store.keys())
        if not self.bm25_ids:
            self.bm25_index = None
            return
        tokenized_corpus = [
            self._tokenize_for_bm25(self.chunk_store[cid].text_for_indexing)
            for cid in self.bm25_ids
        ]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def _save_bm25_store(self) -> None:
        try:
            serializable_data = {
                cid: chunk.model_dump(mode="json") for cid, chunk in self.chunk_store.items()
            }
            with open(BM25_FILE, "w", encoding="utf-8") as f:
                json.dump(serializable_data, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Fehler beim Speichern des BM25-Stores: {e}", exc_info=True)

    def _load_bm25_store(self) -> None:
        if not BM25_FILE.exists():
            if LEGACY_BM25_PICKLE.exists():
                logger.warning(
                    f"Alter BM25-Store {LEGACY_BM25_PICKLE.name} gefunden, wird aus Sicherheitsgründen nicht geladen. "
                    "Bitte einmal /api/v1/rag/sync-from-etl ausführen."
                )
            return
        try:
            with open(BM25_FILE, "r", encoding="utf-8") as f:
                raw_data: Dict[str, dict] = json.load(f)
            self.chunk_store = {
                cid: DocumentChunk.model_validate(c_dict) for cid, c_dict in raw_data.items()
            }
            self._rebuild_bm25_in_memory()
        except Exception as e:
            logger.error(f"Fehler beim Laden des BM25-Stores: {e}", exc_info=True)

    def upsert_chunks(self, chunks: List[DocumentChunk], replace_existing_doc: bool = True) -> int:
        if not chunks:
            return 0

        if replace_existing_doc:
            doc_ids_to_replace = {c.doc_id for c in chunks}
            sources_to_replace = {c.metadata.source_url_or_path for c in chunks}
            self._remove_chunks_by_doc_or_source(doc_ids_to_replace, sources_to_replace)

        texts_to_embed = [c.text_for_indexing for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        embeddings = self.embedder.encode(
            texts_to_embed,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        chroma_metadatas: Any = []
        for c in chunks:
            m = c.metadata
            chroma_metadatas.append(
                {
                    "doc_id": c.doc_id,
                    "title": m.title,
                    "source_url_or_path": m.source_url_or_path,
                    "status": m.status.value,
                    "category": m.category,
                    "district": m.district,
                    "doc_lang": m.doc_lang,
                    "language_level": m.language_level.value,
                    "service_languages_csv": ",".join(m.service_languages),
                    "target_groups_csv": ",".join(t.value for t in m.target_groups),
                    "barrier_free_wheelchair": bool(m.barrier_free_wheelchair),
                    "valid_until": m.valid_until.isoformat() if m.valid_until else "",
                    "is_expired": bool(m.is_expired),
                    "content_hash_sha256": m.content_hash_sha256,
                }
            )
            self.chunk_store[c.chunk_id] = c

        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts_to_embed,
            metadatas=chroma_metadatas,
        )

        self._rebuild_bm25_in_memory()
        self._save_bm25_store()

        logger.info(f"{len(chunks)} Chunks erfolgreich in ChromaDB & BM25 indexiert.")
        return len(chunks)

    def _remove_chunks_by_doc_or_source(self, doc_ids: set, sources: set) -> None:
        to_delete = [
            cid
            for cid, ch in self.chunk_store.items()
            if ch.doc_id in doc_ids or ch.metadata.source_url_or_path in sources
        ]
        if to_delete:
            logger.info(f"§ 10 A c Vektor-Ersetzung: Lösche {len(to_delete)} alte Chunks...")
            try:
                self.collection.delete(ids=to_delete)
            except Exception as e:
                logger.warning(f"ChromaDB Delete Hinweis: {e}")
            for cid in to_delete:
                self.chunk_store.pop(cid, None)

    def run_ttl_maintenance(self) -> Dict[str, int]:
        today = date.today()
        newly_expired = 0
        for chunk in self.chunk_store.values():
            v_until = chunk.metadata.valid_until
            if v_until and v_until < today and not chunk.metadata.is_expired:
                chunk.metadata.is_expired = True
                newly_expired += 1

        if newly_expired > 0:
            self._save_bm25_store()

        return {"total_chunks": len(self.chunk_store), "newly_marked_expired": newly_expired}

    def retrieve(self, req: RetrievalRequest) -> RetrievalResponse:
        if not self.chunk_store:
            return self._build_fallback_response(
                req, "Wissensbasis ist noch leer. Bitte zuerst Dokumente über den ETL-Service indexieren."
            )

        allowed_chunk_ids = self._apply_metadata_filters(req.filters)
        if not allowed_chunk_ids:
            return self._build_fallback_response(
                req,
                "Für die gewählten Filterkriterien (z. B. Stadtteil, Barrierefreiheit oder Sprache) "
                "wurde kein passendes Angebot in der Datenbank gefunden.",
            )

        # 1. Dense Search in ChromaDB (inkl. Cosine-Similarity aus Distanzen)
        query_vec = self.embedder.encode(
            [req.query], normalize_embeddings=True, show_progress_bar=False
        ).tolist()

        n_results = min(req.top_k_candidates * 2, len(self.chunk_store))
        chroma_res = self.collection.query(
            query_embeddings=query_vec,
            n_results=max(n_results, 1),
            include=["distances"],
        )
        raw_dense_ids = (chroma_res.get("ids") or [[]])[0]
        raw_distances = (chroma_res.get("distances") or [[]])[0]

        dense_sim_map: Dict[str, float] = {}
        dense_ranked_ids: List[str] = []
        for cid, dist in zip(raw_dense_ids, raw_distances):
            if cid in allowed_chunk_ids:
                # Cosine-Similarity = 1.0 - Cosine-Distance
                dense_sim_map[cid] = max(0.0, min(1.0, 1.0 - float(dist)))
                if len(dense_ranked_ids) < req.top_k_candidates:
                    dense_ranked_ids.append(cid)

        # 2. Lexikalische BM25-Suche
        bm25_ranked_ids: List[str] = []
        if self.bm25_index and self.bm25_ids:
            query_tokens = self._tokenize_for_bm25(req.query)
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            scored_pairs = [
                (self.bm25_ids[i], float(bm25_scores[i]))
                for i in range(len(self.bm25_ids))
                if self.bm25_ids[i] in allowed_chunk_ids and bm25_scores[i] > 0.0
            ]
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            bm25_ranked_ids = [cid for cid, _ in scored_pairs[: req.top_k_candidates]]

        # 3. Reciprocal Rank Fusion (RRF, k = 60)
        rrf_k = 60
        rrf_scores: Dict[str, float] = {}
        dense_rank_map: Dict[str, int] = {}
        bm25_rank_map: Dict[str, int] = {}

        for rank_idx, cid in enumerate(dense_ranked_ids, start=1):
            dense_rank_map[cid] = rank_idx
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank_idx))

        for rank_idx, cid in enumerate(bm25_ranked_ids, start=1):
            bm25_rank_map[cid] = rank_idx
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank_idx))

        if not rrf_scores:
            for rank_idx, cid in enumerate(list(allowed_chunk_ids)[: req.top_k_candidates], start=1):
                rrf_scores[cid] = 1.0 / (rrf_k + rank_idx)

        fused_candidates = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)[
            : req.top_k_candidates
        ]

        # 4. Cross-Encoder Reranking auf SAUBEREM Text (Titel + text_raw statt Metadaten-Header!)
        cross_inp = [
            [
                req.query,
                f"{self.chunk_store[cid].metadata.title} ({self.chunk_store[cid].metadata.category}): "
                f"{self.chunk_store[cid].text_raw}",
            ]
            for cid in fused_candidates
        ]
        raw_logits = self.reranker.predict(cross_inp, show_progress_bar=False)

        scored_chunks: List[ScoredChunk] = []
        guard_scores: Dict[str, float] = {}  # Relevanz ohne Qualitäts-Boni, nur für den Quellen-Guard
        today = date.today()

        for cid, logit in zip(fused_candidates, raw_logits):
            chunk = self.chunk_store[cid]
            meta = chunk.metadata

            # Normalisierung auf [0.0, 1.0]: genau eine Sigmoid-Funktion, und nur wenn predict() Logits liefert
            if self._reranker_returns_logits:
                rerank_prob = 1.0 / (1.0 + math.exp(-float(logit)))
            else:
                rerank_prob = min(max(float(logit), 0.0), 1.0)
            dense_sim = dense_sim_map.get(cid, 0.5)

            # Kombinierter Relevanz-Score (70 % Cross-Encoder + 30 % BGE-M3 Cosine-Similarity)
            # Verhindert extreme Ausreißer nach unten bei kurzen oder umgangssprachlichen Fragen
            combined_base = (0.70 * rerank_prob) + (0.30 * dense_sim)
            guard_scores[cid] = combined_base

            boost = 0.0
            if meta.status == SourceStatus.OFFICIAL:
                boost += 0.03
            if meta.language_level == LanguageLevel.EASY:
                boost += 0.05
            elif meta.language_level == LanguageLevel.SIMPLE:
                boost += 0.02

            final_score = min(round(combined_base + boost, 4), 1.0)

            ttl_warning = None
            is_expired = meta.is_expired or (meta.valid_until is not None and meta.valid_until < today)
            if is_expired:
                ref_year = (
                    meta.created_or_updated_at.year
                    if meta.created_or_updated_at
                    else (meta.valid_until.year if meta.valid_until else today.year - 1)
                )
                ttl_warning = (
                    f"HINWEIS (§ 10 B c): Diese Quelle hat ihr Gültigkeitsdatum überschritten "
                    f"(Stand {ref_year}). Bitte weise aktiv darauf hin: "
                    f"'Laut einem Dokument aus dem Jahr {ref_year}...'"
                )

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    dense_rank=dense_rank_map.get(cid),
                    bm25_rank=bm25_rank_map.get(cid),
                    rrf_score=round(rrf_scores[cid], 5),
                    reranker_score=round(rerank_prob, 4),
                    final_score=final_score,
                    ttl_warning=ttl_warning,
                )
            )

        scored_chunks.sort(key=lambda sc: sc.final_score, reverse=True)
        top_chunks = scored_chunks[: req.top_k_final]
        # Der Quellen-Guard prüft die Relevanz OHNE die Qualitäts-Boni (offiziell, Leichte Sprache).
        # Die Boni ändern nur die Reihenfolge; sie dürfen keine unpassende Quelle über die Schwelle heben.
        best_score = round(max((guard_scores[sc.chunk.chunk_id] for sc in top_chunks), default=0.0), 4)

        if best_score < req.score_threshold:
            logger.warning(
                f"Quellen-Guard ausgelöst für Anfrage '{req.query}': "
                f"Bester Score ({best_score:.3f}) < Schwellenwert ({req.score_threshold:.3f})."
            )
            return RetrievalResponse(
                query=req.query,
                passed_source_guard=False,
                best_score=best_score,
                score_threshold=req.score_threshold,
                fallback_message=(
                    "Zu dieser Frage liegt mir leider keine gesicherte Information in der "
                    "Quartiersdatenbank vor. Bitte wenden Sie sich direkt an die persönliche "
                    "AWO-Beratungsstelle oder das Quartiersbüro in Hildesheim."
                ),
                results=top_chunks,
            )

        return RetrievalResponse(
            query=req.query,
            passed_source_guard=True,
            best_score=best_score,
            score_threshold=req.score_threshold,
            fallback_message=None,
            results=top_chunks,
        )

    def _apply_metadata_filters(self, filters: Optional[RetrievalFilter]) -> set:
        if not filters:
            return set(self.chunk_store.keys())

        matching_ids = set()
        today = date.today()

        for cid, chunk in self.chunk_store.items():
            m = chunk.metadata

            if filters.only_official_sources and m.status != SourceStatus.OFFICIAL:
                continue
            if filters.require_barrier_free and not m.barrier_free_wheelchair:
                continue
            if filters.exclude_expired and (
                m.is_expired or (m.valid_until is not None and m.valid_until < today)
            ):
                continue
            if filters.category and filters.category.lower() not in m.category.lower():
                continue
            if filters.district and filters.district.lower() not in m.district.lower():
                continue
            if (
                filters.required_service_language
                and filters.required_service_language.lower() not in [s.lower() for s in m.service_languages]
            ):
                continue
            if filters.target_group and filters.target_group not in m.target_groups:
                continue

            matching_ids.add(cid)

        return matching_ids

    def _build_fallback_response(self, req: RetrievalRequest, reason: str) -> RetrievalResponse:
        return RetrievalResponse(
            query=req.query,
            passed_source_guard=False,
            best_score=0.0,
            score_threshold=req.score_threshold,
            fallback_message=(
                f"{reason} Bitte wenden Sie sich direkt an die AWO-Beratung im Quartier Hildesheim."
            ),
            results=[],
        )