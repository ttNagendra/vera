"""
composer.py — LLM composition engine for Vera bot.

Core logic:
  - Routes each trigger.kind to a specialized prompt template
  - Enriches the prompt with all 4 contexts
  - Calls LLM and validates output
  - Re-prompts once if validation fails

Scoring targets (5 dimensions, 0-10 each):
  Specificity         ≥ 8  — anchor on numbers, dates, sources
  Category Fit        ≥ 8  — correct voice per vertical
  Merchant Fit        ≥ 7  — personalized to their data
  Trigger Relevance   ≥ 7  — clearly explains WHY NOW
  Engagement Compulsion ≥ 7 — loss aversion + curiosity + single CTA
"""

import json
import re
from typing import Optional

from llm_client import LLMClient
from validator import extract_json, validate_composed, sanitize_body, normalize_cta


# ---------------------------------------------------------------------------
# Master system prompt (shared across all trigger kinds)
# ---------------------------------------------------------------------------

MASTER_SYSTEM = """You are Vera — magicpin's AI assistant for Indian merchants. You talk to merchants via WhatsApp to help them grow their business.

## YOUR MISSION
Compose ONE WhatsApp message that is highly specific, category-appropriate, and compelling enough that the merchant actually replies.

## THE 8 ENGAGEMENT LEVERS (use 1-2 per message)
1. SPECIFICITY — anchor on a verifiable fact (number, %, date, source). NOT vague claims.
2. LOSS AVERSION — "you're missing X" / "before this window closes"
3. SOCIAL PROOF — "3 dentists in your locality did Y this month"
4. EFFORT EXTERNALIZATION — "I've drafted X — just say go"
5. CURIOSITY — "want to see who?" / "want the full list?"
6. RECIPROCITY — "I noticed Y about your account"
7. ASKING THE MERCHANT — one open question about their business
8. SINGLE BINARY CTA — Reply YES / STOP (not multi-choice unless booking slot)

## VOICE RULES (CRITICAL — violations are penalized)
- Match the merchant's language preference EXACTLY (hi-en mix → use Hinglish naturally)
- Dentists/doctors: clinical peer tone, technical vocabulary OK, NO "guaranteed", NO "cure", NO "best in city"
- Salons: warm, practical, operator-to-operator tone
- Restaurants: peer operator tone, no hype
- Gyms: motivational coaching tone
- Pharmacies: trustworthy, precise, no medical overclaims

## ABSOLUTE RULES (hard violations — judge penalizes -2 to -3 per violation)
- NO URLs in the body (WhatsApp template rules)
- NO generic discount offers like "Flat 30% off" — use service+price format: "Haircut @ ₹99"
- NO long preambles ("I hope you're doing well. I'm reaching out today to...")
- NO multiple CTAs in one message
- NO fabricating data not in the context (no fake peer names, no fake research)
- NO re-introducing yourself after the first message in a conversation
- NO repeating the same message body verbatim
- CTA must land in the LAST sentence

## OUTPUT FORMAT (respond with ONLY this JSON, no other text)
{
  "body": "the WhatsApp message body",
  "cta": "open_ended" | "binary_yes_no" | "binary_confirm_cancel" | "multi_choice_slot" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "copy from trigger context",
  "rationale": "2-3 sentences: why this message, what lever you used, why the CTA fits"
}"""


# ---------------------------------------------------------------------------
# Trigger-kind-specific prompt fragments
# ---------------------------------------------------------------------------

TRIGGER_PROMPTS = {

    "research_digest": """\
TRIGGER TYPE: Research Digest
WHY NOW: A new piece of clinical/industry research just landed that's relevant to this category.
LEVER: Specificity (cite trial_n, %, source) + Curiosity (offer to pull the abstract)
TEMPLATE INTENT: "New research item → specific finding → relevant to their patient/customer mix → want me to help?"
Key: Reference the exact source (journal, issue, page if available). Don't generalize the finding.""",

    "regulation_change": """\
TRIGGER TYPE: Regulatory / Compliance Change
WHY NOW: A regulatory body has updated rules that affect this category.
LEVER: Loss Aversion (deadline, what changes if they ignore it) + Effort Externalization (I'll help you audit)
Key: Cite the authority (DCI, FSSAI, etc.) and the deadline. Be precise, not alarming.""",

    "perf_dip": """\
TRIGGER TYPE: Performance Dip
WHY NOW: One or more KPIs (calls, views, CTR) dropped significantly vs. baseline.
LEVER: Loss Aversion ("you're losing ground") + Specificity (exact delta_pct, metric, window)
TEMPLATE INTENT: "Your [metric] dropped [X%] vs last [window] — here's what I noticed + quick fix"
Key: Compare to peer_stats if available. Don't be alarming — be helpful.""",

    "seasonal_perf_dip": """\
TRIGGER TYPE: Expected Seasonal Dip
WHY NOW: This is a predictable seasonal slow period. The dip is expected — not alarming.
LEVER: Curiosity ("want to see how peers handle this?") + Social Proof
Key: Acknowledge this is normal. Suggest a proactive move before the dip deepens.""",

    "perf_spike": """\
TRIGGER TYPE: Performance Spike
WHY NOW: One or more KPIs spiked above baseline.
LEVER: Reciprocity (sharing a win) + Effort Externalization (help them capitalize on it)
Key: Celebrate briefly, then immediately pivot to "here's the next move to lock it in".""",

    "recall_due": """\
TRIGGER TYPE: Customer Recall Due
WHY NOW: A patient/customer's recall/checkup/service window has opened.
LEVER: Effort Externalization (slots already ready) + Specificity (exact date + price)
SEND AS: merchant_on_behalf (this goes to the CUSTOMER, not the merchant)
Key: Use the customer's name, their language preference, offer specific slot times from the trigger payload.
CTA: multi_choice_slot if slots are provided, otherwise open_ended.""",

    "chronic_refill_due": """\
TRIGGER TYPE: Chronic Prescription Refill Due
WHY NOW: A pharmacy customer's chronic medications are about to run out.
LEVER: Urgency (stock runs out by date) + Effort Externalization (delivery address saved, can order now)
SEND AS: merchant_on_behalf (customer-facing)
Key: List the molecule names, the run-out date, and the convenient ordering path.""",

    "competitor_opened": """\
TRIGGER TYPE: Competitor Opened Nearby
WHY NOW: A competitor just opened within walking distance of this merchant.
LEVER: Loss Aversion + Curiosity ("want to see what they're offering?")
Key: Mention the distance and their price point (if known). Suggest a counter-positioning move.
Don't name the competitor negatively — frame as "a new option in your locality".""",

    "festival_upcoming": """\
TRIGGER TYPE: Festival / Seasonal Opportunity
WHY NOW: A festival is coming up that's relevant to this category.
LEVER: Urgency (days until festival) + Effort Externalization (offer to draft the campaign)
Key: Be category-specific about the festival angle (salons → bridal/party looks; restaurants → catering; pharmacies → gifting).""",

    "ipl_match_today": """\
TRIGGER TYPE: IPL / Sports Match Tonight
WHY NOW: There's a match tonight that could drive footfall or delivery orders.
LEVER: Urgency (match starts in X hours) + Effort Externalization (offer to push a special)
Key: Restaurants and food merchants are most relevant. Tie the offer to watch-party / delivery surge.""",

    "renewal_due": """\
TRIGGER TYPE: Subscription Renewal Due
WHY NOW: This merchant's magicpin subscription expires soon.
LEVER: Loss Aversion (what pauses when it expires: leads, visibility) + Specificity (exact days_remaining)
Key: Frame as "here's what you keep getting vs. what pauses". Don't be pushy — be informative.""",

    "winback_eligible": """\
TRIGGER TYPE: Win-Back (Expired Subscription)
WHY NOW: Merchant's subscription expired N days ago. Performance has dipped since.
LEVER: Loss Aversion + Specificity (exact perf_dip_pct vs. when they were active)
Key: Connect the dip to the expiry. Make the next step frictionless.""",

    "milestone_reached": """\
TRIGGER TYPE: Milestone Reached
WHY NOW: Merchant crossed a meaningful milestone (reviews, customers, orders).
LEVER: Reciprocity (sharing the win) + Curiosity (next milestone, how peers got there faster)
Key: Celebrate specifically (exact number). Immediately propose what to do next.""",

    "review_theme_emerged": """\
TRIGGER TYPE: Review Theme Emerged
WHY NOW: A theme (positive or negative) appeared across multiple recent reviews.
LEVER: Reciprocity (I noticed this for you) + Effort Externalization (I can help address/amplify it)
Key: For negative themes — be constructive, not alarming. For positive — help them amplify.""",

    "curious_ask_due": """\
TRIGGER TYPE: Curiosity Ask (Scheduled)
WHY NOW: A regular "ask the merchant" cadence nudge — no specific event.
LEVER: Asking the Merchant (lever #7 — most underused)
Key: Ask ONE specific, interesting question about their business this week. No offer, no pitch.
Examples: "What's your most-asked service this week?" / "Any service you've been meaning to add?"
CTA: open_ended. No binary ask here.""",

    "gbp_unverified": """\
TRIGGER TYPE: Google Business Profile Unverified
WHY NOW: This merchant's GBP is not verified, costing them significant visibility.
LEVER: Loss Aversion (estimated uplift %) + Effort Externalization (I'll walk them through it)
Key: Lead with the specific estimated_uplift_pct. Make the verification path feel simple.""",

    "dormant_with_vera": """\
TRIGGER TYPE: Dormant — No Recent Conversation
WHY NOW: This merchant hasn't responded to Vera in 14+ days.
LEVER: Curiosity + Reciprocity (something I noticed while they were away)
Key: Warm re-opener. No hard ask. Something interesting they might not know about their own account.
CTA: open_ended.""",

    "active_planning_intent": """\
TRIGGER TYPE: Active Planning — Merchant Has a Question/Intent
WHY NOW: The merchant expressed a specific planning intent in their last message.
LEVER: Effort Externalization ("I've already drafted X") + specifics
Key: ACTION MODE — don't ask qualifying questions. Immediately deliver what they asked for or propose the concrete next step.
Critical: If they said "Yes let's do it" — give them something concrete, not another question.""",

    "supply_alert": """\
TRIGGER TYPE: Supply / Product Alert (Pharmacy)
WHY NOW: A product recall or supply issue affects this pharmacy's inventory.
LEVER: Urgency + Effort Externalization (offer to filter customer list)
Key: Name the exact molecule/product, affected batches (if known). Keep it factual and calm.""",

    "category_seasonal": """\
TRIGGER TYPE: Seasonal Category Trend
WHY NOW: Seasonal demand shift is happening in this category right now.
LEVER: Specificity (exact % trend) + Effort Externalization (shelf/offer action recommended)
Key: Give the specific trend figures, then one concrete action they can take today.""",

    "customer_lapsed_hard": """\
TRIGGER TYPE: Customer Lapsed — Win-Back
WHY NOW: A customer hasn't visited in 57+ days. Win-back window is open.
LEVER: Curiosity + Loss Aversion (for the merchant — lapsed customers cost more to re-acquire)
SEND AS: merchant_on_behalf (goes to the customer)
Key: Reference their previous service/focus. Offer something specific to bring them back.""",

    "trial_followup": """\
TRIGGER TYPE: Trial Session Follow-Up
WHY NOW: A customer did a trial session and hasn't booked a full program yet.
LEVER: Effort Externalization (slot ready) + Social Proof
SEND AS: merchant_on_behalf
Key: Reference the specific trial they did. Offer the logical next step.""",

    "cde_opportunity": """\
TRIGGER TYPE: Continuing Education / Webinar Opportunity
WHY NOW: A relevant professional development opportunity exists.
LEVER: Curiosity + Specificity (credits, fee, speaker, date)
Key: Lead with the tangible benefit (2 CDE credits, free for members). Keep it short.""",

    "wedding_package_followup": """\
TRIGGER TYPE: Bridal / Wedding Package Follow-Up
WHY NOW: A customer had a bridal trial and a key planning window is approaching.
LEVER: Urgency (days_to_wedding) + Effort Externalization
SEND AS: merchant_on_behalf
Key: Reference their wedding date and what needs to happen in the next 30 days. Specific.""",

}

# Default for unknown trigger kinds
DEFAULT_TRIGGER_PROMPT = """\
TRIGGER TYPE: General Merchant Nudge
LEVER: Use the most relevant lever from the trigger payload and merchant signals.
Key: Be specific, be brief, land the CTA at the end."""


# ---------------------------------------------------------------------------
# Context serializers (build clean context strings for the LLM)
# ---------------------------------------------------------------------------

def _fmt_merchant(merchant: dict) -> str:
    identity = merchant.get("identity", {})
    perf = merchant.get("performance", {})
    sub = merchant.get("subscription", {})
    agg = merchant.get("customer_aggregate", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    history = merchant.get("conversation_history", [])
    reviews = merchant.get("review_themes", [])

    lines = [
        f"Merchant: {identity.get('name', 'Unknown')} ({identity.get('city', '?')}, {identity.get('locality', '?')})",
        f"Owner: {identity.get('owner_first_name', 'N/A')} | Verified: {identity.get('verified', False)}",
        f"Languages: {identity.get('languages', ['en'])}",
        f"Subscription: {sub.get('status', 'unknown')} | Plan: {sub.get('plan', 'N/A')} | Days remaining: {sub.get('days_remaining', 'N/A')}",
        f"Performance (30d): views={perf.get('views','?')}, calls={perf.get('calls','?')}, directions={perf.get('directions','?')}, CTR={perf.get('ctr','?')}",
    ]

    delta = perf.get("delta_7d", {})
    if delta:
        lines.append(f"7d delta: views {delta.get('views_pct','?'):+.0%}, calls {delta.get('calls_pct','?'):+.0%}")

    if offers:
        lines.append(f"Active offers: {[o.get('title') for o in offers]}")
    else:
        lines.append("Active offers: none")

    if agg:
        lines.append(f"Customer aggregate: {json.dumps(agg)}")

    if signals:
        lines.append(f"Signals: {signals}")

    if history:
        last = history[-2:] if len(history) >= 2 else history
        lines.append("Recent conversation:")
        for h in last:
            lines.append(f"  [{h.get('from','?')}]: {h.get('body','')[:120]}")

    if reviews:
        pos = [r for r in reviews if r.get("sentiment") == "pos"]
        neg = [r for r in reviews if r.get("sentiment") == "neg"]
        if pos:
            lines.append(f"Positive review themes: {[r.get('theme') for r in pos]}")
        if neg:
            lines.append(f"Negative review themes: {[r.get('theme') for r in neg]}")

    return "\n".join(lines)


def _fmt_category(category: dict) -> str:
    voice = category.get("voice", {})
    peer = category.get("peer_stats", {})
    digest = category.get("digest", [])
    seasonal = category.get("seasonal_beats", [])
    trends = category.get("trend_signals", [])
    offers = category.get("offer_catalog", [])[:5]

    lines = [
        f"Category: {category.get('slug', 'unknown')} ({category.get('display_name', '')})",
        f"Voice tone: {voice.get('tone', 'n/a')} | Register: {voice.get('register', 'n/a')}",
        f"Code-mix: {voice.get('code_mix', 'n/a')}",
        f"Vocab taboos (avoid these words): {voice.get('vocab_taboo', [])}",
        f"Peer stats: avg_ctr={peer.get('avg_ctr','?')}, avg_reviews={peer.get('avg_review_count','?')}, avg_views_30d={peer.get('avg_views_30d','?')}",
        f"Offer catalog (preferred formats): {[o.get('title') for o in offers]}",
    ]

    if digest:
        lines.append("This week's digest items:")
        for d in digest[:3]:
            lines.append(f"  [{d.get('kind','?')}] {d.get('title','')} — {d.get('source','')} | Summary: {d.get('summary','')[:150]}")

    if seasonal:
        lines.append(f"Seasonal beats: {[s.get('note') for s in seasonal[:3]]}")

    if trends:
        trend_strs = ["{} {:+.0%}".format(t.get('query',''), t.get('delta_yoy', 0)) for t in trends[:3]]
        lines.append(f"Trend signals: {trend_strs}")

    return "\n".join(lines)


def _fmt_trigger(trigger: dict, category: dict) -> str:
    payload = trigger.get("payload", {})

    # Resolve top_item_id reference from category digest
    if "top_item_id" in payload:
        item_id = payload["top_item_id"]
        for d in category.get("digest", []):
            if d.get("id") == item_id:
                payload = {**payload, "resolved_digest_item": d}
                break

    lines = [
        f"Trigger kind: {trigger.get('kind', 'unknown')}",
        f"Scope: {trigger.get('scope', '?')} | Source: {trigger.get('source', '?')}",
        f"Urgency: {trigger.get('urgency', '?')}/5",
        f"Suppression key: {trigger.get('suppression_key', '')}",
        f"Expires at: {trigger.get('expires_at', 'N/A')}",
        f"Payload: {json.dumps(payload, indent=2)}",
    ]
    return "\n".join(lines)


def _fmt_customer(customer: dict) -> str:
    identity = customer.get("identity", {})
    rel = customer.get("relationship", {})
    prefs = customer.get("preferences", {})

    lines = [
        f"Customer: {identity.get('name', 'Unknown')} | Language: {identity.get('language_pref', 'en')}",
        f"Age band: {identity.get('age_band', 'N/A')}",
        f"State: {customer.get('state', 'unknown')}",
        f"First visit: {rel.get('first_visit', 'N/A')} | Last visit: {rel.get('last_visit', 'N/A')}",
        f"Total visits: {rel.get('visits_total', 0)} | Services received: {rel.get('services_received', [])[-3:]}",
        f"Lifetime value: ₹{rel.get('lifetime_value', 0):,}",
        f"Preferred slots: {prefs.get('preferred_slots', 'N/A')}",
        f"Consent scope: {customer.get('consent', {}).get('scope', [])}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

class VeraComposer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def compose(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict] = None,
        conversation_history: Optional[list] = None,
        intent_mode: str = "pitching",
        action_hint: Optional[str] = None,   # "auto_nudge" | "deflect"
    ) -> dict:
        """
        Compose a WhatsApp message from the 4 contexts.
        Returns a dict with: body, cta, send_as, suppression_key, rationale
        """
        trigger_kind = trigger.get("kind", "unknown")
        trigger_prompt = TRIGGER_PROMPTS.get(trigger_kind, DEFAULT_TRIGGER_PROMPT)

        system = MASTER_SYSTEM
        user = self._build_user_prompt(
            category, merchant, trigger, customer,
            trigger_prompt, conversation_history,
            intent_mode, action_hint
        )

        # --- First attempt ---
        raw = await self.llm.complete(system, user)
        composed = extract_json(raw)

        if not composed:
            # --- Re-prompt with strict JSON-only instruction ---
            raw = await self.llm.complete(
                system,
                user + "\n\nIMPORTANT: Your previous response was not valid JSON. "
                       "Reply with ONLY the JSON object, no other text."
            )
            composed = extract_json(raw)

        if not composed:
            # Fallback — minimal safe response
            return self._fallback(trigger, merchant, customer)

        # --- Normalize and clean ---
        composed["cta"] = normalize_cta(composed.get("cta", "open_ended"))
        composed["send_as"] = composed.get("send_as", "vera")
        if composed["send_as"] not in {"vera", "merchant_on_behalf"}:
            composed["send_as"] = "merchant_on_behalf" if customer else "vera"
        if not composed.get("suppression_key"):
            composed["suppression_key"] = trigger.get("suppression_key", f"auto:{trigger.get('id','')}")

        # --- Sanitize URLs (never hard-fail, just remove) ---
        if "body" in composed:
            from validator import URL_PATTERN
            if URL_PATTERN.search(composed["body"]):
                composed["body"] = sanitize_body(composed["body"])

        return composed

    def _build_user_prompt(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict],
        trigger_prompt: str,
        conversation_history: Optional[list],
        intent_mode: str,
        action_hint: Optional[str],
    ) -> str:

        parts = [
            "## TRIGGER CONTEXT",
            trigger_prompt,
            "",
            "## CATEGORY CONTEXT",
            _fmt_category(category),
            "",
            "## MERCHANT CONTEXT",
            _fmt_merchant(merchant),
            "",
            "## TRIGGER PAYLOAD",
            _fmt_trigger(trigger, category),
        ]

        if customer:
            parts += ["", "## CUSTOMER CONTEXT", _fmt_customer(customer)]

        if conversation_history:
            parts += ["", "## CONVERSATION SO FAR (last 4 turns)"]
            for t in conversation_history[-4:]:
                parts.append(f"  [{t.get('role','?')}]: {t.get('message','')[:200]}")

        if intent_mode == "action":
            parts += [
                "",
                "## IMPORTANT — ACTION MODE",
                "The merchant has explicitly committed. DO NOT ask qualifying questions.",
                "Give them the concrete next step or deliverable immediately.",
            ]

        if action_hint == "auto_nudge":
            parts += [
                "",
                "## CONTEXT — AUTO-REPLY DETECTED",
                "The merchant's last reply looks like a WhatsApp Business auto-reply.",
                "Write a very short message acknowledging this and flagging it for the owner.",
                "Something like: 'Looks like your auto-reply is on 😊 Owner ko dikhna chahiye — just reply YES when you see this.'",
                "Keep it light, warm, under 2 sentences. CTA: binary_yes_no.",
            ]

        if action_hint == "deflect":
            parts += [
                "",
                "## CONTEXT — OUT-OF-SCOPE QUESTION",
                "The merchant asked something outside Vera's scope (e.g., GST, legal, etc.).",
                "Politely decline THAT request, then immediately pivot back to the original trigger topic.",
                "Under 2 sentences for the decline. Then 1 sentence to re-engage the original topic.",
            ]

        parts += [
            "",
            "## YOUR TASK",
            "Compose the next WhatsApp message. Return ONLY the JSON object. No preamble, no explanation.",
        ]

        return "\n".join(parts)

    def _fallback(self, trigger: dict, merchant: dict, customer: Optional[dict]) -> dict:
        """Minimal safe fallback if LLM completely fails."""
        mid = trigger.get("merchant_id", "")
        merchant_name = merchant.get("identity", {}).get("name", "there")
        kind = trigger.get("kind", "update")
        return {
            "body": f"Hi {merchant_name}, quick check-in from Vera — wanted to share something about your {kind.replace('_', ' ')} update. Got a minute?",
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf" if customer else "vera",
            "suppression_key": trigger.get("suppression_key", f"fallback:{mid}"),
            "rationale": "Fallback response — LLM composition failed. Minimal safe message sent.",
        }

    async def compose_reply(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict],
        conversation_turns: list,
        merchant_message: str,
        intent_mode: str = "pitching",
        action_hint: Optional[str] = None,
    ) -> dict:
        """
        Compose a reply to a merchant/customer message in an ongoing conversation.
        """
        system = MASTER_SYSTEM + "\n\nYou are CONTINUING an existing conversation. DO NOT re-introduce yourself."

        turn_history = "\n".join([
            f"  [{t.get('role','?')}]: {t.get('message','')[:200]}"
            for t in conversation_turns[-6:]
        ])

        trigger_kind = trigger.get("kind", "unknown")
        trigger_prompt = TRIGGER_PROMPTS.get(trigger_kind, DEFAULT_TRIGGER_PROMPT)

        user_parts = [
            "## ORIGINAL TRIGGER",
            trigger_prompt,
            "",
            "## CATEGORY CONTEXT",
            _fmt_category(category),
            "",
            "## MERCHANT CONTEXT",
            _fmt_merchant(merchant),
        ]

        if customer:
            user_parts += ["", "## CUSTOMER CONTEXT", _fmt_customer(customer)]

        user_parts += [
            "",
            "## CONVERSATION HISTORY",
            turn_history,
            "",
            f"## MERCHANT'S LATEST MESSAGE",
            f'"{merchant_message}"',
        ]

        if intent_mode == "action":
            user_parts += [
                "",
                "CRITICAL — ACTION MODE: Merchant committed. Give the concrete deliverable NOW. No qualifying questions.",
            ]

        if action_hint == "auto_nudge":
            user_parts += [
                "",
                "Auto-reply detected — short, warm, flag for owner. Under 2 sentences.",
            ]
        elif action_hint == "deflect":
            user_parts += [
                "",
                "Out-of-scope question — politely decline then pivot back to original topic.",
            ]

        user_parts += [
            "",
            "Compose Vera's next reply. Return ONLY JSON.",
        ]

        user = "\n".join(user_parts)
        raw = await self.llm.complete(system, user)
        composed = extract_json(raw)

        if not composed:
            raw = await self.llm.complete(system, user + "\n\nReturn ONLY valid JSON, nothing else.")
            composed = extract_json(raw)

        if not composed:
            return {
                "body": "Got it — let me look into that and get back to you shortly.",
                "cta": "none",
                "send_as": "merchant_on_behalf" if customer else "vera",
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": "Fallback reply — LLM failed.",
            }

        composed["cta"] = normalize_cta(composed.get("cta", "open_ended"))
        if not composed.get("send_as"):
            composed["send_as"] = "merchant_on_behalf" if customer else "vera"
        if not composed.get("suppression_key"):
            composed["suppression_key"] = trigger.get("suppression_key", "")

        from validator import URL_PATTERN
        if "body" in composed and URL_PATTERN.search(composed["body"]):
            composed["body"] = sanitize_body(composed["body"])

        return composed
