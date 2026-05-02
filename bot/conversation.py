"""
conversation.py — Multi-turn conversation state manager for Vera bot.

Handles:
  - Turn history per conversation_id
  - Auto-reply pattern detection (graceful exit after 3 attempts)
  - Intent transition (qualifying → action mode)
  - Hostile / opt-out detection
  - Suppression registry (per-conversation and per-merchant)
"""

import re
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pattern libraries
# ---------------------------------------------------------------------------

# WhatsApp Business auto-reply fingerprints (case-insensitive)
AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"aapki jaankari ke liye.*shukriya",
    r"aapki madad ke liye.*shukriya",
    r"main.*automated.*assistant",
    r"i am an automated",
    r"our team will (respond|get back)",
    r"hamari team.*pahuncha",
    r"we will respond (shortly|soon)",
    r"jald hi.*sampark",
]

# Explicit opt-out / hostility
HOSTILE_PATTERNS = [
    r"\bstop\b",
    r"\bnot interested\b",
    r"\bspam\b",
    r"don'?t (message|contact|send|text)",
    r"leave me alone",
    r"remove (me|my number)",
    r"unsubscribe",
    r"band karo",
    r"mat bhejo",
    r"irritating",
    r"useless",
    r"bothering me",
]

# Explicit commitment / intent transition
COMMITMENT_PATTERNS = [
    r"\blet'?s do it\b",
    r"\bgo ahead\b",
    r"\byes\b.*\bproceed\b",
    r"\bproceed\b",
    r"\bconfirm\b",
    r"\bwhat'?s next\b",
    r"\bok\b.*\bsend\b",
    r"\bsend it\b",
    r"\bdo it\b",
    r"\bagree\b",
    r"\bchalo\b",
    r"\btheek hai\b.*\bbhejo\b",
    r"\bkaro\b",
    r"\byes please\b",
    r"\byes, go\b",
    r"\byes go\b",
]

# Out-of-scope deflection triggers
OUT_OF_SCOPE_PATTERNS = [
    r"\bgst\b",
    r"\btax (filing|return)\b",
    r"\blegal\b",
    r"\bloan\b",
    r"\binsurance\b",
    r"\baccounting\b",
]


def _matches_any(text: str, patterns: List[str]) -> bool:
    t = text.lower().strip()
    return any(re.search(p, t) for p in patterns)


# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------

class Turn:
    def __init__(self, role: str, message: str, action: Optional[str] = None):
        self.role = role        # "vera" | "merchant" | "customer"
        self.message = message
        self.action = action    # for vera turns: "send" | "wait" | "end"
        self.ts = datetime.now(timezone.utc).isoformat()
        self.is_auto_reply = False


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

class ConversationState:
    """State for a single conversation_id."""

    def __init__(self, conversation_id: str, merchant_id: str,
                 customer_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.turns: List[Turn] = []
        self.suppressed = False         # This conversation is dead
        self.intent_mode = "pitching"   # "pitching" | "action"
        self.auto_reply_count = 0
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_turn(self, role: str, message: str, action: Optional[str] = None) -> Turn:
        t = Turn(role, message, action)
        if role in ("merchant", "customer"):
            t.is_auto_reply = _matches_any(message, AUTO_REPLY_PATTERNS)
        self.turns.append(t)
        return t

    def last_vera_body(self) -> Optional[str]:
        """Return the most recent message body Vera sent."""
        for t in reversed(self.turns):
            if t.role == "vera":
                return t.message
        return None

    def vera_sent_bodies(self) -> List[str]:
        return [t.message for t in self.turns if t.role == "vera"]

    def consecutive_auto_replies(self) -> int:
        """Count consecutive auto-replies from the end of the turn list."""
        count = 0
        for t in reversed(self.turns):
            if t.role in ("merchant", "customer") and t.is_auto_reply:
                count += 1
            elif t.role in ("merchant", "customer"):
                break   # real reply resets the chain
        return count


# ---------------------------------------------------------------------------
# Conversation manager (global registry)
# ---------------------------------------------------------------------------

class ConversationManager:
    """Global registry of all active conversations."""

    def __init__(self):
        self._convs: Dict[str, ConversationState] = {}
        self._suppressed_merchants: set = set()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lookup / creation
    # ------------------------------------------------------------------

    def get_or_create(self, conversation_id: str, merchant_id: str,
                      customer_id: Optional[str] = None) -> ConversationState:
        with self._lock:
            if conversation_id not in self._convs:
                self._convs[conversation_id] = ConversationState(
                    conversation_id, merchant_id, customer_id
                )
            return self._convs[conversation_id]

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        with self._lock:
            return self._convs.get(conversation_id)

    def suppress_merchant(self, merchant_id: str):
        with self._lock:
            self._suppressed_merchants.add(merchant_id)

    def is_merchant_suppressed(self, merchant_id: str) -> bool:
        with self._lock:
            return merchant_id in self._suppressed_merchants

    # ------------------------------------------------------------------
    # Reply analysis — decide the next move
    # ------------------------------------------------------------------

    def analyze_reply(self, conv: ConversationState,
                      message: str) -> Tuple[str, dict]:
        """
        Given the merchant's reply message, decide what action to take.

        Returns: (action_type, metadata)
          action_type: "send" | "wait" | "end" | "continue"
          metadata: {
            "reason": str,
            "is_auto_reply": bool,
            "is_hostile": bool,
            "is_commitment": bool,
            "is_out_of_scope": bool,
            "consecutive_auto_replies": int,
          }
        """
        is_auto = _matches_any(message, AUTO_REPLY_PATTERNS)
        is_hostile = _matches_any(message, HOSTILE_PATTERNS)
        is_commitment = _matches_any(message, COMMITMENT_PATTERNS)
        is_oos = _matches_any(message, OUT_OF_SCOPE_PATTERNS)

        # Add this turn to history
        conv.add_turn("merchant", message)
        consecutive_auto = conv.consecutive_auto_replies()

        meta = {
            "is_auto_reply": is_auto,
            "is_hostile": is_hostile,
            "is_commitment": is_commitment,
            "is_out_of_scope": is_oos,
            "consecutive_auto_replies": consecutive_auto,
        }

        # --- Hostile / opt-out → immediate END ---
        if is_hostile:
            conv.suppressed = True
            self.suppress_merchant(conv.merchant_id)
            return "end", {**meta, "reason": "Merchant opted out or expressed hostility. Closing gracefully."}

        # --- Auto-reply detection ---
        if is_auto:
            if consecutive_auto == 1:
                # First auto-reply: try once more with an explicit nudge
                return "send_auto_nudge", {**meta,
                    "reason": "Detected auto-reply (first time). Sending one explicit nudge for owner."}
            elif consecutive_auto == 2:
                # Second auto-reply: back off
                return "wait", {**meta,
                    "reason": "Same auto-reply twice. Owner not at phone. Waiting 4h before retry.",
                    "wait_seconds": 14400}
            else:
                # Third+ auto-reply: end
                conv.suppressed = True
                return "end", {**meta,
                    "reason": "Auto-reply 3× in a row. Zero engagement signal. Closing conversation."}

        # --- Intent transition → ACTION mode ---
        if is_commitment:
            conv.intent_mode = "action"
            return "continue", {**meta,
                "reason": "Merchant committed. Switching to action mode."}

        # --- Out of scope → deflect ---
        if is_oos:
            return "deflect", {**meta,
                "reason": "Out-of-scope question. Politely redirect."}

        # --- Normal reply → continue conversation ---
        return "continue", {**meta,
            "reason": "Normal engaged reply. Composing follow-up."}
