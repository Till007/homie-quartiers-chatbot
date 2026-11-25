"""
server_runtime/build_bundle.py
Exportiert alle qualitätsgeprüften Chunks aus data/processed_chunks/
komplett OFFLINE (ohne API-Key & ohne Kosten) in ein kompaktes Read-Only-Wissenspaket
(server_runtime/knowledge_bundle.json) für den Laptop- oder vServer-Betrieb.

Aufruf im Hauptordner:
    python server_runtime/build_bundle.py
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROCESSED_CHUNKS_DIR = Path("data/processed_chunks")
OUTPUT_BUNDLE_PATH = Path("server_runtime/knowledge_bundle.json")


def main() -> None:
    if not PROCESSED_CHUNKS_DIR.exists():
        print(f"FEHLER: Ordner '{PROCESSED_CHUNKS_DIR}' nicht gefunden.")
        sys.exit(1)

    all_chunks: List[Dict[str, Any]] = []
    seen_ids = set()

    for json_file in sorted(PROCESSED_CHUNKS_DIR.glob("*.json")):
        try:
            raw_list = json.loads(json_file.read_text(encoding="utf-8"))
            if not isinstance(raw_list, list):
                continue
            for item in raw_list:
                cid = item.get("chunk_id")
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                meta = item.get("metadata", {})
                all_chunks.append(
                    {
                        "chunk_id": cid,
                        "doc_id": item.get("doc_id", ""),
                        "text_raw": item.get("text_raw", ""),
                        "text_for_indexing": item.get("text_for_indexing", ""),
                        "metadata": {
                            "title": meta.get("title", ""),
                            "publisher": meta.get("publisher"),
                            "source_url_or_path": meta.get("source_url_or_path", ""),
                            "status": meta.get("status", "unofficial"),
                            "category": meta.get("category", "Allgemein"),
                            "district": meta.get("district", "Hildesheim-Gesamt"),
                            "language_level": meta.get("language_level", "standard"),
                            "service_languages": meta.get("service_languages", ["de"]),
                            "valid_until": meta.get("valid_until"),
                            "is_expired": bool(meta.get("is_expired", False)),
                            "facility_name": meta.get("facility_name"),
                            "address": meta.get("address"),
                            "phone": meta.get("phone"),
                            "opening_hours": meta.get("opening_hours"),
                        },
                    }
                )
        except Exception as e:
            print(f"Warnung: Überspringe {json_file.name}: {e}")

    if not all_chunks:
        print("FEHLER: Keine Chunks in data/processed_chunks/ gefunden.")
        sys.exit(1)

    bundle = {
        "bundle_version": "2.0-offline-hybrid",
        "chunk_count": len(all_chunks),
        "chunks": all_chunks,
    }
    OUTPUT_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_BUNDLE_PATH.write_text(
        json.dumps(bundle, ensure_ascii=False), encoding="utf-8"
    )
    size_kb = OUTPUT_BUNDLE_PATH.stat().st_size / 1024
    print(f"Fertig! {len(all_chunks)} Chunks verpackt unter: {OUTPUT_BUNDLE_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()