"""
Regex-first input extraction and validation utilities.

Layer 1 rule:
- Regex runs first on every field.
- LLM extraction is only a fallback.
- Anything extracted by the LLM must be re-validated by this module.
"""

from __future__ import annotations

import calendar
import datetime
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from config import EXTRACTABLE_FIELDS, GROQ_API_KEY, GROQ_MODEL


ACCOUNT_ID_RE = re.compile(r"\bACC\d{4,}\b")
DOB_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
AADHAAR_LAST4_RE = re.compile(
    r"(?:aadhaar|aadhar|adhar|last\s*4|last\s*four)[^\d]{0,20}(\d{4})",
    re.IGNORECASE,
)
PINCODE_RE = re.compile(
    r"(?:pincode|pin\s*code|pin|postal\s*code|zip)[^\d]{0,20}(\d{6})",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(
    r"(?<!\d)(?:₹|rs\.?|inr|\$)?\s*([0-9]+(?:,[0-9]{2,3})*(?:\.\d{1,2})?|[0-9]+(?:\.\d{1,2})?)(?!\d)",
    re.IGNORECASE,
)
CARD_NUMBER_RE = re.compile(
    r"(?:card\s*number|card|number)?[^\d]{0,20}((?:\d[ -]?){12,19})",
    re.IGNORECASE,
)
CVV_RE = re.compile(r"(?:cvv|cvc|security\s*code)[^\d]{0,20}(\d{3,4})", re.IGNORECASE)
EXPIRY_RE = re.compile(
    r"(?:exp(?:iry|iration)?|valid\s*thru|valid\s*through)?[^\d]{0,20}"
    r"(?:(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2}|\d{4})|"
    r"(\d{4})\s*[/-]\s*(0?[1-9]|1[0-2]))",
    re.IGNORECASE,
)
NAME_LABEL_RE = re.compile(
    r"(?:full\s*name|name|my\s*name\s*is|i\s*am|i'm)\s*(?:is|:)?\s*([A-Z][A-Za-z.' -]{1,80})",
    re.IGNORECASE,
)
CARDHOLDER_RE = re.compile(
    r"(?:cardholder\s*name|card\s*holder\s*name|name\s*on\s*card|cardholder|card\s*holder)"
    r"\s*(?:is|:)?\s*([A-Za-z][A-Za-z.' -]{1,80})",
    re.IGNORECASE,
)

def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _only_digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value or ""))


def _title_like_candidate(value: str) -> Optional[str]:
    cleaned = _clean_text(value)
    cleaned = re.sub(r"[.,;:!?]+$", "", cleaned).strip()

    if not cleaned:
        return None

    lowered = cleaned.lower()
    bad_fragments = {
        "account",
        "aadhaar",
        "aadhar",
        "pincode",
        "pin",
        "dob",
        "date",
        "amount",
        "card",
        "cvv",
        "expiry",
        "pay",
        "payment",
    }

    if any(fragment in lowered for fragment in bad_fragments):
        return None

    if not re.fullmatch(r"[A-Za-z][A-Za-z.' -]{1,80}", cleaned):
        return None

    return cleaned


def validate_account_id(value: Any) -> Optional[str]:
    if value is None:
        return None

    candidate = str(value).strip().upper()
    if re.fullmatch(r"ACC\d{4,}", candidate):
        return candidate

    return None


def validate_full_name(value: Any) -> Optional[str]:
    if value is None:
        return None

    candidate = _title_like_candidate(str(value))
    if candidate and len(candidate.split()) >= 2:
        return candidate

    return None


def validate_dob(value: Any) -> Optional[str]:
    """
    Validate DOB in YYYY-MM-DD format.

    Uses calendar/date math so leap years are handled correctly:
    - 1988-02-29 is valid.
    - 1990-02-29 is invalid.
    """
    if value is None:
        return None

    candidate = str(value).strip()
    if not re.fullmatch(r"(?:19|20)\d{2}-\d{2}-\d{2}", candidate):
        return None

    year_s, month_s, day_s = candidate.split("-")
    year = int(year_s)
    month = int(month_s)
    day = int(day_s)

    if month < 1 or month > 12:
        return None

    _, max_day = calendar.monthrange(year, month)
    if day < 1 or day > max_day:
        return None

    return f"{year:04d}-{month:02d}-{day:02d}"


def validate_aadhaar_last4(value: Any) -> Optional[str]:
    if value is None:
        return None

    candidate = _only_digits(value)
    if re.fullmatch(r"\d{4}", candidate):
        return candidate

    return None


def validate_pincode(value: Any) -> Optional[str]:
    if value is None:
        return None

    candidate = _only_digits(value)
    if re.fullmatch(r"\d{6}", candidate):
        return candidate

    return None


def normalize_amount(value: Any) -> Optional[float]:
    if value is None:
        return None

    candidate = str(value).strip().replace(",", "")
    candidate = re.sub(r"^(₹|rs\.?|inr|\$)\s*", "", candidate, flags=re.IGNORECASE)

    try:
        amount = Decimal(candidate)
    except InvalidOperation:
        return None

    if amount <= 0:
        return None

    if abs(amount.as_tuple().exponent) > 2:
        return None

    return float(amount)


def validate_amount(value: Any) -> Optional[float]:
    return normalize_amount(value)


def strip_spaces_from_card(value: Any) -> str:
    return re.sub(r"[\s-]", "", str(value or ""))


def luhn_check(card_number: Any) -> bool:
    """
    Pure Python Luhn implementation.

    Returns True only if the supplied card number passes the Luhn checksum.
    """
    digits = strip_spaces_from_card(card_number)

    if not digits.isdigit():
        return False

    if len(digits) < 12 or len(digits) > 19:
        return False

    total = 0
    should_double = False

    for char in reversed(digits):
        digit = ord(char) - ord("0")

        if should_double:
            digit *= 2
            if digit > 9:
                digit -= 9

        total += digit
        should_double = not should_double

    return total % 10 == 0


def validate_card_number(value: Any) -> Optional[str]:
    if value is None:
        return None

    candidate = strip_spaces_from_card(value)

    if not candidate.isdigit():
        return None

    if len(candidate) < 12 or len(candidate) > 19:
        return None

    if set(candidate) == {"0"}:
        return None

    if not luhn_check(candidate):
        return None

    return candidate


def _is_amex(card_number: Optional[str]) -> bool:
    if not card_number:
        return False

    digits = strip_spaces_from_card(card_number)
    return len(digits) == 15 and digits.startswith(("34", "37"))


def validate_cvv(value: Any, card_number: Optional[str] = None) -> Optional[str]:
    if value is None:
        return None

    candidate = _only_digits(value)

    if _is_amex(card_number):
        return candidate if re.fullmatch(r"\d{4}", candidate) else None

    return candidate if re.fullmatch(r"\d{3}", candidate) else None


def _normalize_expiry_year(year: int) -> int:
    if year < 100:
        return 2000 + year
    return year


def validate_expiry(month: Any, year: Any) -> Optional[tuple[int, int]]:
    try:
        month_i = int(str(month).strip())
        year_i = _normalize_expiry_year(int(str(year).strip()))
    except (TypeError, ValueError):
        return None

    if month_i < 1 or month_i > 12:
        return None

    today = datetime.date.today()
    _, last_day = calendar.monthrange(year_i, month_i)
    expiry_date = datetime.date(year_i, month_i, last_day)

    if expiry_date < today:
        return None

    return month_i, year_i


def mask_card_number(value: Any) -> Optional[str]:
    digits = strip_spaces_from_card(value)

    if not digits:
        return None

    last4 = digits[-4:] if len(digits) >= 4 else digits
    return f"**** **** **** {last4}"


def remove_sensitive_fields(data: dict[str, Any]) -> dict[str, Any]:
    safe = dict(data)

    for key in ["dob", "aadhaar_last4", "pincode", "cvv"]:
        safe.pop(key, None)

    if safe.get("card_number"):
        safe["card_number"] = mask_card_number(safe["card_number"])

    return safe


def extract_account_id(text: str) -> Optional[str]:
    match = ACCOUNT_ID_RE.search(text or "")
    return validate_account_id(match.group(0)) if match else None


def extract_dob(text: str) -> Optional[str]:
    for match in DOB_RE.finditer(text or ""):
        valid = validate_dob(match.group(0))
        if valid:
            return valid

    return None


def extract_aadhaar_last4(text: str) -> Optional[str]:
    text = text or ""

    labeled = AADHAAR_LAST4_RE.search(text)
    if labeled:
        return validate_aadhaar_last4(labeled.group(1))

    cleaned = _clean_text(text)

    # Only treat a raw 4-digit message as Aadhaar last 4 if the whole message is exactly 4 digits.
    # This prevents expiry years like 2027 or DOB years like 1990 from being misread as Aadhaar.
    if re.fullmatch(r"\d{4}", cleaned):
        return validate_aadhaar_last4(cleaned)

    return None


def extract_pincode(text: str) -> Optional[str]:
    text = text or ""

    labeled = PINCODE_RE.search(text)
    if labeled:
        return validate_pincode(labeled.group(1))

    six_digit_values = re.findall(r"\b\d{6}\b", text)
    if len(six_digit_values) == 1:
        return validate_pincode(six_digit_values[0])

    return None


def extract_amount(text: str) -> Optional[float]:
    text = text or ""

    amount_context = re.search(
        r"(?:pay|payment|amount|balance|₹|rs\.?|inr|\$)[^\d]{0,20}"
        r"([0-9]+(?:,[0-9]{2,3})*(?:\.\d{1,2})?|[0-9]+(?:\.\d{1,2})?)",
        text,
        re.IGNORECASE,
    )

    raw_amount = None

    if amount_context:
        raw_amount = amount_context.group(1)
    else:
        standalone = _clean_text(text).replace(",", "")
        if re.fullmatch(r"[0-9]+(?:\.\d{1,2})?", standalone):
            raw_amount = standalone

    if raw_amount is None:
        return None

    try:
        return float(str(raw_amount).replace(",", ""))
    except ValueError:
        return None


def extract_card_number(text: str) -> Optional[str]:
    text = text or ""

    for match in CARD_NUMBER_RE.finditer(text):
        raw = match.group(1) if match.lastindex else match.group(0)
        candidate = validate_card_number(raw)
        if candidate:
            return candidate

    return None


def extract_cvv(text: str, card_number: Optional[str] = None) -> Optional[str]:
    text = text or ""

    labeled = CVV_RE.search(text)
    if labeled:
        return validate_cvv(labeled.group(1), card_number=card_number)

    return None


# def extract_expiry(text: str) -> tuple[Optional[int], Optional[int]]:
#     text = text or ""

#     # Prefer labeled expiry patterns first.
#     labeled = re.search(
#         r"(?:exp(?:iry|iration)?|valid\s*thru|valid\s*through)[^\d]{0,20}"
#         r"(?:(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2}|\d{4})|"
#         r"(\d{4})\s*[/-]\s*(0?[1-9]|1[0-2]))",
#         text,
#         re.IGNORECASE,
#     )

#     candidates = []

#     if labeled:
#         candidates.append(labeled)

#     # Also allow bare MM/YYYY or MM/YY when present in a payment sentence.
#     for match in re.finditer(
#         r"(?:(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2}|\d{4})|"
#         r"(\d{4})\s*[/-]\s*(0?[1-9]|1[0-2]))",
#         text,
#         re.IGNORECASE,
#     ):
#         candidates.append(match)

#     for match in candidates:
#         if match.group(1) and match.group(2):
#             month = match.group(1)
#             year = match.group(2)
#         else:
#             year = match.group(3)
#             month = match.group(4)

#         valid = validate_expiry(month, year)
#         if valid:
#             return valid

#     return None, None
def extract_expiry(text: str) -> tuple[Optional[int], Optional[int]]:
    text = text or ""

    patterns = [
        # expiry 12/2027, exp 12-27, valid thru 12/2027
        r"(?:exp(?:iry|iration)?|valid\s*thru|valid\s*through)\D{0,20}"
        r"(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2}|\d{4})",

        # plain 12/2027 or 12-27
        r"\b(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2}|\d{4})\b",

        # plain 12 2027
        r"\b(0?[1-9]|1[0-2])\s+(\d{2}|\d{4})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        month = match.group(1)
        year = match.group(2)

        valid = validate_expiry(month, year)
        if valid:
            return valid

    return None, None


def extract_full_name(text: str) -> Optional[str]:
    text = text or ""

    # Do not treat cardholder name as identity full_name.
    if re.search(r"\b(cardholder|card\s*holder|name\s*on\s*card)\b", text, re.IGNORECASE):
        return None

    # Safer pattern for "my name is X" and "I am X"
    labeled = re.search(
        r"(?:full\s*name|my\s*name\s*is|name\s*is|i\s*am|i'm)\s*"
        r"([A-Z][A-Za-z.' -]{1,80})",
        text,
        re.IGNORECASE,
    )

    if labeled:
        raw = labeled.group(1)

        # Stop name extraction before other fields / sentence connectors.
        raw = re.split(
            r"\b(?:and|dob|date\s*of\s*birth|aadhaar|aadhar|pincode|pin\s*code|pay|payment|amount|card)\b",
            raw,
            flags=re.IGNORECASE,
        )[0]

        valid = validate_full_name(raw.strip(" ,.-"))
        if valid:
            return valid

    cleaned = _clean_text(text)

    # Only treat the entire message as a name if it is just words.
    if re.fullmatch(r"[A-Za-z][A-Za-z.' -]{1,80}", cleaned):
        valid = validate_full_name(cleaned)
        if valid:
            return valid

    return None


def extract_cardholder_name(text: str) -> Optional[str]:
    text = text or ""

    labeled = CARDHOLDER_RE.search(text)
    if labeled:
        raw = labeled.group(1)

        # Stop before other payment fields if they appear in the same sentence.
        raw = re.split(
            r"\b(?:card\s*number|number|cvv|cvc|expiry|expiration|valid\s*thru|valid\s*through)\b",
            raw,
            flags=re.IGNORECASE,
        )[0]

        valid = validate_full_name(raw.strip(" ,.-"))
        if valid:
            return valid

    return None


def extract_fields_regex_first(text: str) -> dict[str, Any]:
    """
    Extract every supported field using deterministic regex and validators only.
    """
    card_number = extract_card_number(text)
    expiry_month, expiry_year = extract_expiry(text)

    fields = {
        "account_id": extract_account_id(text),
        "full_name": extract_full_name(text),
        "dob": extract_dob(text),
        "aadhaar_last4": extract_aadhaar_last4(text),
        "pincode": extract_pincode(text),
        "amount": extract_amount(text),
        "cardholder_name": extract_cardholder_name(text),
        "card_number": card_number,
        "cvv": extract_cvv(text, card_number=card_number),
        "expiry_month": expiry_month,
        "expiry_year": expiry_year,
    }

    return {key: value for key, value in fields.items() if value is not None}


def _safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            return {}

        try:
            parsed = json.loads(json_match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def extract_fields_with_groq(text: str) -> dict[str, Any]:
    """
    Fallback LLM extraction using the official groq Python package.

    This function must only be called after regex extraction has run.
    The returned values are not trusted until validate_extracted_fields() runs.
    """
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_key_here":
        return {}

    try:
        from groq import Groq
    except ImportError:
        return {}

    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = (
        "Extract payment collection fields from the user's message. "
        "Return ONLY strict JSON. Do not explain. "
        "Use null for missing fields. "
        "Fields: account_id, full_name, dob, aadhaar_last4, pincode, amount, "
        "cardholder_name, card_number, cvv, expiry_month, expiry_year. "
        "DOB must be YYYY-MM-DD. Do not infer unknown values."
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text or ""},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception:
        return {}

    content = completion.choices[0].message.content or "{}"
    parsed = _safe_json_loads(content)

    return {
        key: value
        for key, value in parsed.items()
        if key in EXTRACTABLE_FIELDS and value not in (None, "", [])
    }


def validate_extracted_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """
    Re-validate all extracted fields.

    This function is used for both regex output and LLM fallback output.
    Invalid fields are silently dropped rather than trusted.
    """
    clean: dict[str, Any] = {}

    account_id = validate_account_id(fields.get("account_id"))
    if account_id:
        clean["account_id"] = account_id

    full_name = validate_full_name(fields.get("full_name"))
    if full_name:
        clean["full_name"] = full_name

    dob = validate_dob(fields.get("dob"))
    if dob:
        clean["dob"] = dob

    aadhaar_last4 = validate_aadhaar_last4(fields.get("aadhaar_last4"))
    if aadhaar_last4:
        clean["aadhaar_last4"] = aadhaar_last4

    pincode = validate_pincode(fields.get("pincode"))
    if pincode:
        clean["pincode"] = pincode

    if "amount" in fields and fields.get("amount") is not None:
        try:
            clean["amount"] = float(fields.get("amount"))
        except (TypeError, ValueError):
            pass

    cardholder_name = validate_full_name(fields.get("cardholder_name"))
    if cardholder_name:
        clean["cardholder_name"] = cardholder_name

    card_number = validate_card_number(fields.get("card_number"))
    if card_number:
        clean["card_number"] = card_number

    cvv = validate_cvv(fields.get("cvv"), card_number=card_number)
    if cvv:
        clean["cvv"] = cvv

    expiry = validate_expiry(fields.get("expiry_month"), fields.get("expiry_year"))
    if expiry:
        clean["expiry_month"], clean["expiry_year"] = expiry

    return clean


def extract_fields(text: str, use_llm_fallback: bool = True) -> dict[str, Any]:
    """
    Public extraction entry point.

    Regex always runs first. Groq is used only as fallback for fields that regex
    failed to extract. LLM values are re-validated before returning.
    """
    regex_fields = validate_extracted_fields(extract_fields_regex_first(text))

    if not use_llm_fallback:
        return regex_fields

    missing_fields = [field for field in EXTRACTABLE_FIELDS if field not in regex_fields]
    if not missing_fields:
        return regex_fields

    llm_fields = validate_extracted_fields(extract_fields_with_groq(text))

    merged = dict(regex_fields)
    for field in missing_fields:
        if field in llm_fields:
            merged[field] = llm_fields[field]

    return merged