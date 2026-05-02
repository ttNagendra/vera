"""
main.py — Vera Bot: FastAPI server exposing all 5 judge endpoints.

Endpoints:
  GET  /v1/healthz   — liveness probe
  GET  /v1/metadata  — bot identity
  POST /v1/context   — receive context push (category/merchant/customer/trigger)
  POST /v1/tick      — bot decides to send proactive messages
  POST /v1/reply     — handle merchant reply, return next move

Usage:
  cp .env.example .env       # fill in LLM_API_KEY
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8080
"""

import os
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from context_store import ContextStore
from conversation import ConversationManager
from llm_client import create_llm_client
from composer import VeraComposer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vera")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(title="Vera Bot", version=os.getenv("BOT_VERSION", "1.0.0"))
START_TIME = time.time()

ctx_store = ContextStore()
conv_manager = ConversationManager()

# LLM client — initialized lazily on first request to avoid startup crash
_llm_client = None
_composer = None

def get_composer() -> VeraComposer:
    global _llm_client, _composer
    if _composer is None:
        _llm_client = create_llm_client()
        _composer = VeraComposer(_llm_client)
        log.info(f"LLM provider: {_llm_client.provider_name}")
    return _composer

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 1


# ---------------------------------------------------------------------------
# Trigger prioritization
# ---------------------------------------------------------------------------

# Higher urgency = picked first. Within same urgency, prefer kinds that
# score better on the 5-dimension rubric.
KIND_PRIORITY = {
    "supply_alert": 10,
    "regulation_change": 9,
    "active_planning_intent": 9,
    "renewal_due": 8,
    "perf_dip": 8,
    "chronic_refill_due": 8,
    "recall_due": 7,
    "competitor_opened": 7,
    "review_theme_emerged": 6,
    "winback_eligible": 6,
    "milestone_reached": 6,
    "perf_spike": 5,
    "research_digest": 5,
    "customer_lapsed_hard": 5,
    "trial_followup": 5,
    "wedding_package_followup": 5,
    "festival_upcoming": 4,
    "ipl_match_today": 4,
    "category_seasonal": 4,
    "cde_opportunity": 3,
    "gbp_unverified": 3,
    "seasonal_perf_dip": 3,
    "curious_ask_due": 2,
    "dormant_with_vera": 2,
}


def score_trigger(trigger: dict) -> int:
    urgency = trigger.get("urgency", 1)
    kind_score = KIND_PRIORITY.get(trigger.get("kind", ""), 1)
    return urgency * 2 + kind_score


def select_best_trigger(trigger_ids: list[str]) -> Optional[str]:
    """Pick the highest-priority trigger from the list."""
    best_id = None
    best_score = -1
    for tid in trigger_ids:
        t = ctx_store.get_trigger(tid)
        if not t:
            continue
        s = score_trigger(t)
        if s > best_score:
            best_score = s
            best_id = tid
    return best_id


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def _build_turn_history(conv_state) -> list[dict]:
    return [
        {"role": t.role, "message": t.message}
        for t in conv_state.turns
    ]


def _resolve_contexts(trigger: dict):
    """Resolve category, merchant, customer from a trigger."""
    merchant_id = trigger.get("merchant_id", "")
    customer_id = trigger.get("customer_id")

    merchant = ctx_store.get_merchant(merchant_id) or {}
    category_slug = merchant.get("category_slug", "")
    category = ctx_store.get_category(category_slug) or {}
    customer = ctx_store.get_customer(customer_id) if customer_id else None

    return category, merchant, customer, merchant_id, customer_id


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "bot": "Vera",
        "status": "online",
        "message": "Welcome to Vera Merchant AI Assistant! API endpoints are available at /v1/*",
        "endpoints": ["/v1/healthz", "/v1/metadata", "/v1/context", "/v1/tick", "/v1/reply"]
    }

@app.get("/v1/healthz")
async def healthz():
    counts = ctx_store.counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": os.getenv("TEAM_NAME", "Team Vera"),
        "team_members": [m.strip() for m in os.getenv("TEAM_MEMBERS", "Alice").split(",")],
        "model": get_composer().llm.provider_name if _llm_client else os.getenv("LLM_PROVIDER", "gemini"),
        "approach": (
            "4-context composition with trigger-kind dispatch. "
            "25+ specialized prompt templates per trigger kind. "
            "Auto-reply detection (pattern match, 3-turn escalation). "
            "Intent-transition detection for action mode. "
            "Post-LLM validation: CTA normalization, URL removal, anti-repetition."
        ),
        "contact_email": os.getenv("CONTACT_EMAIL", "team@example.com"),
        "version": os.getenv("BOT_VERSION", "1.0.0"),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    result = ctx_store.put(body.scope, body.context_id, body.version, body.payload)

    if result.get("accepted"):
        log.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")
        return result
    else:
        # Version conflict → 409, invalid scope → 400
        reason = result.get("reason", "unknown")
        status = 409 if reason == "stale_version" else 400
        return JSONResponse(content=result, status_code=status)


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Judge calls this periodically. Bot selects the best trigger and composes a message.
    Returns actions[] — empty list is valid (restraint is rewarded).
    """
    if not body.available_triggers:
        return {"actions": []}

    # Pick the single highest-priority trigger this tick
    best_tid = select_best_trigger(body.available_triggers)
    if not best_tid:
        return {"actions": []}

    trigger = ctx_store.get_trigger(best_tid)
    if not trigger:
        return {"actions": []}

    category, merchant, customer, merchant_id, customer_id = _resolve_contexts(trigger)

    if not merchant:
        log.warning(f"No merchant found for trigger {best_tid}")
        return {"actions": []}

    # Check merchant suppression
    if conv_manager.is_merchant_suppressed(merchant_id):
        log.info(f"Merchant {merchant_id} suppressed — skipping")
        return {"actions": []}

    # Compose message
    try:
        composer = get_composer()
        composed = await composer.compose(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )
    except Exception as e:
        log.error(f"Composition failed for {best_tid}: {e}", exc_info=True)
        return {"actions": []}

    # Build conversation_id and register it
    conv_id = f"conv_{merchant_id}_{best_tid}"
    conv = conv_manager.get_or_create(conv_id, merchant_id, customer_id)
    conv.add_turn("vera", composed.get("body", ""), action="send")

    action = {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed.get("send_as", "vera"),
        "trigger_id": best_tid,
        "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
        "template_params": _extract_template_params(composed, merchant, customer),
        "body": composed.get("body", ""),
        "cta": composed.get("cta", "open_ended"),
        "suppression_key": composed.get("suppression_key", trigger.get("suppression_key", "")),
        "rationale": composed.get("rationale", ""),
    }

    log.info(f"Action composed for {merchant_id} | trigger={best_tid} | cta={action['cta']}")
    return {"actions": [action]}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Judge sends a merchant/customer reply. Bot must respond within 30s.
    Returns: {action: "send"|"wait"|"end", body?, cta?, rationale}
    """
    conv_id = body.conversation_id
    merchant_id = body.merchant_id or ""
    customer_id = body.customer_id
    message = body.message

    # Get or create conversation state
    conv = conv_manager.get_or_create(conv_id, merchant_id, customer_id)

    # Guard — suppressed conversation
    if conv.suppressed:
        return {
            "action": "end",
            "rationale": "Conversation was previously suppressed. Not re-engaging.",
        }

    # Analyze the merchant's reply
    action_type, meta = conv_manager.analyze_reply(conv, message)

    log.info(f"Reply received [{conv_id}] turn={body.turn_number} | action_type={action_type}")

    # --- END cases ---
    if action_type == "end":
        return {
            "action": "end",
            "rationale": meta.get("reason", "Closing conversation."),
        }

    # --- WAIT case ---
    if action_type == "wait":
        wait_s = meta.get("wait_seconds", 14400)
        return {
            "action": "wait",
            "wait_seconds": wait_s,
            "rationale": meta.get("reason", "Backing off."),
        }

    # --- Find trigger for this conversation (best available) ---
    # Try to find the original trigger from the conversation ID
    trigger = _find_trigger_for_conv(conv_id, merchant_id)
    if not trigger:
        # No trigger context — return action-type-specific responses
        if action_type == "send_auto_nudge":
            return {
                "action": "send",
                "body": "Looks like your auto-reply is on! Owner ko dikhna chahiye — just reply YES when you see this.",
                "cta": "binary_yes_no",
                "rationale": "Auto-reply detected. Sending a short nudge to reach the owner directly.",
            }
        if action_type == "continue" and meta.get("is_commitment"):
            return {
                "action": "send",
                "body": "Bilkul! Aapka profile update draft kar rahi hoon — ek minute mein bhejti hoon. Confirm karein?",
                "cta": "binary_confirm_cancel",
                "rationale": "Merchant committed. Switching to action mode with a concrete next step.",
            }
        return {
            "action": "send",
            "body": "Got it — I'll pull up the details and get back to you right away.",
            "cta": "none",
            "rationale": "No trigger context available for this conversation.",
        }

    category, merchant, customer, _, _ = _resolve_contexts(trigger)
    if not merchant:
        return {
            "action": "send",
            "body": "Understood — I'll follow up on this shortly.",
            "cta": "none",
            "rationale": "No merchant context found.",
        }

    # Determine action hint for composer
    action_hint = None
    if action_type == "send_auto_nudge":
        action_hint = "auto_nudge"
    elif action_type == "deflect":
        action_hint = "deflect"

    # --- Compose reply ---
    try:
        composer = get_composer()
        turn_history = _build_turn_history(conv)

        # Check anti-repetition
        previous_bodies = conv.vera_sent_bodies()

        composed = await composer.compose_reply(
            category=category,
            merchant=merchant,
            trigger=trigger,
            customer=customer,
            conversation_turns=turn_history,
            merchant_message=message,
            intent_mode=conv.intent_mode,
            action_hint=action_hint,
        )
    except Exception as e:
        log.error(f"Reply composition failed [{conv_id}]: {e}", exc_info=True)
        return {
            "action": "send",
            "body": "Let me check that and get back to you.",
            "cta": "none",
            "rationale": "Composition error — minimal fallback.",
        }

    # Register Vera's reply in conversation history
    body_text = composed.get("body", "")
    conv.add_turn("vera", body_text, action="send")

    return {
        "action": "send",
        "body": body_text,
        "cta": composed.get("cta", "open_ended"),
        "rationale": composed.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_trigger_for_conv(conv_id: str, merchant_id: str) -> Optional[dict]:
    """
    Try to find the trigger for a conversation by:
    1. Parsing the conv_id (format: conv_{merchant_id}_{trigger_id})
    2. Searching all triggers for this merchant
    """
    # Try to extract trigger_id from conv_id
    prefix = f"conv_{merchant_id}_"
    if conv_id.startswith(prefix):
        trigger_id = conv_id[len(prefix):]
        t = ctx_store.get_trigger(trigger_id)
        if t:
            return t

    # Fallback: find any active trigger for this merchant
    for t in ctx_store.all_triggers():
        if t.get("merchant_id") == merchant_id:
            return t

    return None


def _extract_template_params(composed: dict, merchant: dict, customer: Optional[dict]) -> list:
    """
    Extract 3 template params from the composed message for WhatsApp template format.
    Template: {{1}} = name, {{2}} = hook, {{3}} = CTA
    """
    body = composed.get("body", "")
    sentences = [s.strip() for s in body.split(".") if s.strip()]

    name = (customer or merchant).get("identity", {}).get("name", "")
    hook = sentences[0] if sentences else body[:80]
    cta_text = sentences[-1] if len(sentences) > 1 else ""

    return [name, hook, cta_text]


# ---------------------------------------------------------------------------
# Optional teardown endpoint (judge calls at end of test)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    """Wipe all state at end of test (per testing brief §11)."""
    ctx_store._store.clear()
    conv_manager._convs.clear()
    conv_manager._suppressed_merchants.clear()
    log.info("Teardown complete — all state wiped")
    return {"status": "torn_down"}


# ---------------------------------------------------------------------------
# CORS for dashboard
# ---------------------------------------------------------------------------

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
