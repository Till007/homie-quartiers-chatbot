"""
services/orchestrator_service/memory.py
Implementiert das flüchtige Dialoggedächtnis ausschließlich im Arbeitsspeicher (RAM)
mit automatischem Session-Reset nach 60 Sekunden Inaktivität gemäß TA-06 (DSGVO Privacy-by-Design).
"""

import time
from typing import Dict, List, Tuple
from shared.logger import get_service_logger
from shared.schemas import ChatMessage

logger = get_service_logger("orchestrator-memory")


class EphemeralSessionMemory:
    """
    Reiner In-Memory-Speicher ohne Festplatten-Persistenz.
    Löscht Sitzungen automatisch nach Ablauf von ttl_seconds (Standard: 60 Sekunden).
    """

    def __init__(self, ttl_seconds: int = 60, max_turns: int = 6):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        # Struktur: session_id -> (last_active_monotonic_time, List[ChatMessage])
        self._store: Dict[str, Tuple[float, List[ChatMessage]]] = {}

    def _cleanup_expired(self) -> None:
        """Entfernt alle Sessions aus dem RAM, die länger als ttl_seconds inaktiv waren."""
        now = time.monotonic()
        expired_sessions = [
            sid for sid, (last_ts, _) in self._store.items() if (now - last_ts) > self.ttl_seconds
        ]
        for sid in expired_sessions:
            self._store.pop(sid, None)
            logger.info(
                f"[DSGVO TA-06] Sitzung '{sid}' nach >{self.ttl_seconds}s Inaktivität restlos aus dem RAM gelöscht."
            )

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Gibt die letzten Gesprächsrunden zurück (falls nicht abgelaufen)."""
        self._cleanup_expired()
        if session_id not in self._store:
            return []
        last_ts, messages = self._store[session_id]
        # Aktualisiere Zeitstempel bei Zugriff
        self._store[session_id] = (time.monotonic(), messages)
        return list(messages)

    def append_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Speichert Frage und Antwort im RAM und begrenzt die Historie auf max_turns."""
        self._cleanup_expired()
        _, history = self._store.get(session_id, (time.monotonic(), []))
        history.append(ChatMessage(role="user", content=user_msg))
        history.append(ChatMessage(role="assistant", content=assistant_msg))

        # Behalte nur die letzten max_turns Nachrichtenpaare
        if len(history) > self.max_turns * 2:
            history = history[-(self.max_turns * 2) :]

        self._store[session_id] = (time.monotonic(), history)

    def reset_session(self, session_id: str) -> bool:
        """Manueller Sofort-Reset (z. B. über 'Gespräch beenden'-Button am Kiosk-Terminal)."""
        existed = session_id in self._store
        self._store.pop(session_id, None)
        logger.info(f"[DSGVO TA-06] Manueller Session-Reset für '{session_id}' durchgeführt.")
        return existed