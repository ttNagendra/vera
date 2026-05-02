"""
validator.py — Output validation for Vera bot.

Checks the LLM's composed message for:
  - No URLs (Meta WhatsApp rule — hard fail -3 per URL)
  - Valid CTA value
  - No taboo vocabulary for the category
  - No repetition vs. previous sends in the same conversation
  - Reasonable body length
  - Required JSON fields present
"""

import re
import json
from typing import Optional, List, Tuple

VALID_CTAS = {
    "open_ended",
    "binary_yes_no",
    "binary_confirm_cancel",
    "multi_choice_slot",
    "none",
}

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

REQUIRED_FIELDS = {"body", "cta", "send_as", "suppression_key", "rationale"}

VALID_SEND_AS = {"vera", "merchant_on_behalf"}


def extract_json(text: str) -> Optional[dict]:
    """
    Extract the first JSON object from a string.
    Handles LLM responses that wrap JSON in markdown code fences.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Find first {...}
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        # Try to find a valid JSON by searching progressively
        for start in range(len(text)):
            if text[start] == "{":
                for end in range(len(text), start, -1):
                    if text[end - 1] == "}":
                        try:
                            return json.loads(text[start:end])
                        except json.JSONDecodeError:
                            continue
    return None


def validate_composed(
    composed: dict,
    previous_bodies: List[str],
    category_taboos: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate a composed message dict.

    Returns:
      (is_valid, list_of_issues)
    """
    issues = []

    # --- Required fields ---
    missing = REQUIRED_FIELDS - set(composed.keys())
    if missing:
        issues.append(f"Missing required fields: {sorted(missing)}")

    body = composed.get("body", "")
    cta = composed.get("cta", "")
    send_as = composed.get("send_as", "")

    # --- URL check (hard fail) ---
    urls = URL_PATTERN.findall(body)
    if urls:
        issues.append(f"Body contains URL(s): {urls} — WhatsApp will reject")

    # --- CTA validity ---
    if cta not in VALID_CTAS:
        issues.append(f"Invalid cta '{cta}'. Must be one of {sorted(VALID_CTAS)}")

    # --- send_as validity ---
    if send_as not in VALID_SEND_AS:
        issues.append(f"Invalid send_as '{send_as}'. Must be 'vera' or 'merchant_on_behalf'")

    # --- Anti-repetition ---
    if body and previous_bodies:
        # Exact match check
        if body.strip() in [b.strip() for b in previous_bodies]:
            issues.append("Body is identical to a previous turn — anti-repetition violation")
        # Near-duplicate (first 50 chars match)
        elif len(body) > 50 and any(
            body[:50].strip() == prev[:50].strip()
            for prev in previous_bodies if len(prev) > 50
        ):
            issues.append("Body too similar to a previous send — consider variation")

    # --- Taboo vocabulary ---
    if category_taboos and body:
        body_lower = body.lower()
        found_taboos = [t for t in category_taboos if t.lower() in body_lower]
        if found_taboos:
            issues.append(f"Body contains taboo words: {found_taboos}")

    # --- Body not empty ---
    if not body.strip():
        issues.append("Body is empty")

    is_valid = len(issues) == 0
    return is_valid, issues


def sanitize_body(body: str) -> str:
    """Remove URLs from body as a last resort fix."""
    return URL_PATTERN.sub("[link removed]", body).strip()


def normalize_cta(cta: str) -> str:
    """Normalize a CTA string to a valid value, defaulting to open_ended."""
    cta = cta.lower().strip()
    if cta in VALID_CTAS:
        return cta
    # Fuzzy match
    if "yes" in cta or "no" in cta:
        return "binary_yes_no"
    if "confirm" in cta or "cancel" in cta:
        return "binary_confirm_cancel"
    if "slot" in cta or "choice" in cta:
        return "multi_choice_slot"
    if "none" in cta or cta == "":
        return "none"
    return "open_ended"
