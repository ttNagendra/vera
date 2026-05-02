"""
context_store.py — Versioned in-memory context storage for Vera bot.

Stores category, merchant, customer, and trigger contexts pushed by the judge.
Thread-safe with version conflict detection (idempotent by scope+context_id+version).
"""

import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple


class ContextStore:
    """
    Thread-safe versioned store for all 4 context types.
    Key: (scope, context_id) → {version, payload, stored_at}
    """

    VALID_SCOPES = {"category", "merchant", "customer", "trigger"}

    def __init__(self):
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        """
        Store a context. Returns:
          - {"accepted": True, "ack_id": ..., "stored_at": ...} on success
          - {"accepted": False, "reason": "stale_version", "current_version": N} on conflict
          - {"accepted": False, "reason": "invalid_scope"} on bad scope
        """
        if scope not in self.VALID_SCOPES:
            return {"accepted": False, "reason": "invalid_scope",
                    "details": f"scope must be one of {sorted(self.VALID_SCOPES)}"}

        key = (scope, context_id)
        stored_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            existing = self._store.get(key)

            if existing is not None and existing["version"] >= version:
                # Idempotent re-push of same version is a no-op (409)
                return {
                    "accepted": False,
                    "reason": "stale_version",
                    "current_version": existing["version"]
                }

            self._store[key] = {
                "version": version,
                "payload": payload,
                "stored_at": stored_at,
            }

        ack_id = f"ack_{context_id}_v{version}".replace(" ", "_")
        return {"accepted": True, "ack_id": ack_id, "stored_at": stored_at}

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        """Return the payload dict, or None if not found."""
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["payload"] if entry else None

    def get_merchant(self, merchant_id: str) -> Optional[dict]:
        return self.get("merchant", merchant_id)

    def get_category(self, slug: str) -> Optional[dict]:
        return self.get("category", slug)

    def get_customer(self, customer_id: str) -> Optional[dict]:
        return self.get("customer", customer_id)

    def get_trigger(self, trigger_id: str) -> Optional[dict]:
        return self.get("trigger", trigger_id)

    # ------------------------------------------------------------------
    # Counts (for /v1/healthz)
    # ------------------------------------------------------------------

    def counts(self) -> dict:
        """Return count of stored contexts per scope."""
        result = {s: 0 for s in self.VALID_SCOPES}
        with self._lock:
            for (scope, _) in self._store:
                if scope in result:
                    result[scope] += 1
        return result

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def all_triggers(self) -> list:
        """Return all trigger payloads."""
        with self._lock:
            return [
                v["payload"] for (scope, _), v in self._store.items()
                if scope == "trigger"
            ]

    def all_merchants(self) -> list:
        """Return all merchant payloads."""
        with self._lock:
            return [
                v["payload"] for (scope, _), v in self._store.items()
                if scope == "merchant"
            ]
