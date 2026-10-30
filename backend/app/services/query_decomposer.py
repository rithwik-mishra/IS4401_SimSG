"""
backend/app/services/query_decomposer.py
-----------------------------------------
Decomposes an unstructured user pitch into structured concept attributes for SimSG.
Extracts:
  - concept_name: str
  - concept_description: str
  - target_planning_area: Literal["Ang Mo Kio", "Tampines", "Bukit Timah", "Jurong East", "Bukit Merah", "Outram"]
  - candidate_price_sgd: float (default 4.50 if omitted)
  - category_type: Literal["beverage", "bakery", "full_service_dining", "retail"]

Supports user overrides:
  - override_price: Optional[float]
  - override_town: Optional[str]

Model:
  - Uses `gpt-4o-mini` with structured JSON parsing when OPEN_API_KEY / OPENAI_API_KEY is available.
  - Automatically falls back to deterministic rule-based/regex extraction on API errors,
    network failures, or missing credentials so offline tests and dry-runs never crash.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load environment variables
# Check backend/.env explicitly, then fallback to environment
# ---------------------------------------------------------------------------
_CURRENT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _CURRENT_DIR.parent.parent  # backend/
_ENV_PATH = _BACKEND_DIR / ".env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)
else:
    load_dotenv()

# ---------------------------------------------------------------------------
# Constants & Allowed Values
# ---------------------------------------------------------------------------
ALLOWED_PLANNING_AREAS = [
    "Ang Mo Kio",
    "Tampines",
    "Bukit Timah",
    "Jurong East",
    "Bukit Merah",
    "Outram",
]
DEFAULT_PLANNING_AREA = "Ang Mo Kio"
DEFAULT_PRICE_SGD = 4.50

PlanningAreaType = Literal[
    "Ang Mo Kio",
    "Tampines",
    "Bukit Timah",
    "Jurong East",
    "Bukit Merah",
    "Outram",
]

CategoryType = Literal[
    "beverage",
    "bakery",
    "full_service_dining",
    "retail",
]

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class DecomposedConcept(BaseModel):
    """Structured attributes decomposed from a business concept pitch."""

    concept_name: str = Field(
        ...,
        description="Concise name or title of the business concept (e.g. 'Artisanal Single-Origin Kopi Bar')",
    )
    concept_description: str = Field(
        ...,
        description="Operational description and core offerings of the concept",
    )
    target_planning_area: PlanningAreaType = Field(
        default="Ang Mo Kio",
        description="Singapore planning area where the concept will be deployed",
    )
    candidate_price_sgd: float = Field(
        default=4.50,
        description=(
            "Candidate price in Singapore Dollars (SGD) for the primary product or offering "
            "(e.g. 7.0 for 'S$7', 4.5 for '$4.50'). Look for price indicators such as 'S$', '$', 'SGD', "
            "'dollars', 'bucks', or numeric prices in the text. Do NOT default to 4.50 if an explicit price is mentioned anywhere in the query."
        ),
    )
    category_type: CategoryType = Field(
        ...,
        description="One of: 'beverage', 'bakery', 'full_service_dining', 'retail'",
    )


# Alias for compatibility with external references
DecomposedQuery = DecomposedConcept


# ---------------------------------------------------------------------------
# Deterministic Rule-Based Extraction Fallback
# ---------------------------------------------------------------------------
_AREA_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("Ang Mo Kio", re.compile(r"\b(ang\s+mo\s+kio|amk|angmokio)\b", re.IGNORECASE)),
    ("Tampines", re.compile(r"\b(tampines|tamp)\b", re.IGNORECASE)),
    ("Bukit Timah", re.compile(r"\b(bukit\s+timah|bt\s+timah)\b", re.IGNORECASE)),
    ("Jurong East", re.compile(r"\b(jurong\s+east|jurong|je)\b", re.IGNORECASE)),
    ("Bukit Merah", re.compile(r"\b(bukit\s+merah|redhill|tiong\s+bahru|bm)\b", re.IGNORECASE)),
    ("Outram", re.compile(r"\b(outram|outram\s+park|chinatown|tanjong\s+pagar)\b", re.IGNORECASE)),
]

_PRICE_PATTERNS: List[re.Pattern[str]] = [
    # 1. Prefixed currency: S$7, S$ 7, $7, SGD 7, SGD7, S$7.50
    re.compile(r"(?:s\$|\$|sgd\s*)\s*(\d+(?:\.\d{1,2})?)", re.IGNORECASE),
    # 2. Postfixed currency: 7 SGD, 7.50 SGD, 7 dollars, 7 bucks, 7-dollar
    re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:-|\s)*(?:sgd|dollars?|bucks)", re.IGNORECASE),
    # 3. Contextual verbs: priced at 7, selling for $7, costs 7.50, at $7, for $7, price 7
    re.compile(r"\b(?:priced\s+at|selling\s+for|price\s+(?:is|at)?|costs?|for|at)\s+(?:s\$|\$|sgd\s*)?\s*(\d+(?:\.\d{1,2})?)\b", re.IGNORECASE),
    # 4. Per-unit / per-serving notation: 7/cup, 7 / cup, 7/pax, 7 per pax, 7 each, 7/bowl
    re.compile(r"\b(?:s\$|\$|sgd\s*)?\s*(\d+(?:\.\d{1,2})?)\s*(?:/\s*(?:cup|pax|bowl|plate|set|item|unit|person)|per\s+(?:cup|pax|bowl|plate|set|item|unit|person)|each)\b", re.IGNORECASE),
    # 5. Nett / tax notation: S$7 nett, $7++, 7 nett
    re.compile(r"\b(?:s\$|\$|sgd\s*)?\s*(\d+(?:\.\d{1,2})?)\s*(?:nett|\+\+)\b", re.IGNORECASE),
]

_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "bakery": [
        "bakery", "bake", "bread", "pastry", "pastries", "croissant", "croissants",
        "sourdough", "toast", "kaya toast", "cake", "cakes", "tart", "tarts",
        "patisserie", "confectionery", "bagel", "bagels", "donut", "donuts",
        "doughnut", "doughnuts", "bun", "buns", "bakehouse", "waffle", "waffles"
    ],
    "beverage": [
        "kopi", "coffee", "cafe", "café", "tea", "teh", "boba", "bubble tea",
        "latte", "espresso", "bar", "brew", "brewer", "brewery", "drink", "drinks",
        "smoothie", "smoothies", "juice", "juices", "beverage", "beverages",
        "kombucha", "matcha", "cocktail", "cocktails", "pour-over", "pourover",
        "cold brew", "coldbrew", "milk tea"
    ],
    "full_service_dining": [
        "restaurant", "dining", "noodle", "noodles", "ramen", "chicken rice",
        "rice", "bistro", "hotpot", "steak", "pasta", "buffet", "hawker",
        "food court", "cuisine", "eatery", "diner", "lunch", "dinner", "supper",
        "bbq", "barbecue", "western food", "zi char", "tze char", "dim sum",
        "burger", "burgers", "pizza", "sushi"
    ],
    "retail": [
        "retail", "shop", "store", "bookstore", "stationery", "clothing",
        "apparel", "florist", "gift", "grocery", "groceries", "boutique",
        "supermarket", "mart", "goods", "merchandise", "lifestyle", "accessories"
    ],
}


def _extract_area_rule_based(text: str) -> str:
    """Matches text against Singapore planning areas, defaulting to Ang Mo Kio."""
    for area_name, pattern in _AREA_PATTERNS:
        if pattern.search(text):
            return area_name
    return DEFAULT_PLANNING_AREA


def _extract_explicit_price(text: str) -> Optional[float]:
    """
    Extracts an explicitly stated price from text if any price pattern matches.
    Returns None if no price pattern is found in the text (unlike _extract_price_rule_based which defaults).
    """
    for pattern in _PRICE_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                val = float(match.group(1))
                if val > 0:
                    return round(val, 2)
            except (ValueError, IndexError):
                continue
    return None


def _extract_price_rule_based(text: str) -> float:
    """Extracts candidate price in SGD, defaulting to DEFAULT_PRICE_SGD (4.50)."""
    explicit = _extract_explicit_price(text)
    if explicit is not None:
        return explicit
    return DEFAULT_PRICE_SGD


def _extract_category_rule_based(text: str) -> CategoryType:
    """Categorizes pitch using keyword matching frequencies."""
    text_lower = text.lower()
    scores: Dict[CategoryType, int] = {
        "beverage": 0,
        "bakery": 0,
        "full_service_dining": 0,
        "retail": 0,
    }

    for cat, kws in _CATEGORY_KEYWORDS.items():
        for kw in kws:
            # Word boundary matching
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                scores[cat] += 1  # type: ignore[index]

    best_cat, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_cat

    # Fallback heuristic
    if any(w in text_lower for w in ["eat", "food", "meal", "dine"]):
        return "full_service_dining"
    if any(w in text_lower for w in ["drink", "sip", "cup"]):
        return "beverage"
    if any(w in text_lower for w in ["buy", "sell", "merch"]):
        return "retail"
    return "beverage"


def _extract_concept_name_rule_based(text: str) -> str:
    """Generates a clean concept title from the unstructured text."""
    # Check for quotes: "Artisanal Single-Origin Kopi Bar"
    quote_match = re.search(r'["\']([^"\']{3,60})["\']', text)
    if quote_match:
        return quote_match.group(1).strip()

    cleaned = text.strip()
    changed = True
    while changed:
        prev = cleaned
        # Strip conversational prefixes
        cleaned = re.sub(
            r"^(?:i\s+want\s+to\s+open|we\s+are\s+launching|opening\s+a|launching\s+a|a\s+new|introducing|planning\s+to\s+open|an|a)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        # Strip leading price tag (e.g. S$7, $7, 7 SGD, 7 dollars)
        cleaned = re.sub(
            r"^(?:(?:s\$|\$|sgd\s*)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:sgd|dollars?|bucks))\s*(?:nett|\+\+)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        changed = (cleaned != prev)

    # Cut off at common boundary prepositions
    parts = re.split(r"\b(?:in|at|selling|priced\s+at|for|with)\b|[,\.\?!;]", cleaned, flags=re.IGNORECASE)
    candidate = parts[0].strip()
    words = candidate.split()
    if 1 <= len(words) <= 8:
        candidate_title = " ".join(words).title()
        if len(candidate_title) >= 3:
            return candidate_title

    # Fallback to first 5 words
    first_words = " ".join(cleaned.split()[:5]).title()
    return first_words if len(first_words) >= 3 else "Singapore Retail Concept"


def decompose_query_rule_based(
    raw_query: str,
    override_price: Optional[float] = None,
    override_town: Optional[str] = None,
) -> DecomposedConcept:
    """
    Deterministic rule-based extractor that runs without external API dependencies.
    Guarantees that test suites and offline dry-runs never crash.
    """
    extracted_town = _extract_area_rule_based(raw_query)
    extracted_price = _extract_price_rule_based(raw_query)
    extracted_cat = _extract_category_rule_based(raw_query)
    concept_name = _extract_concept_name_rule_based(raw_query)
    concept_desc = raw_query.strip()

    # Apply overrides
    final_town = _normalize_town(override_town) if override_town else extracted_town
    final_price = float(override_price) if override_price is not None else extracted_price

    return DecomposedConcept(
        concept_name=concept_name,
        concept_description=concept_desc,
        target_planning_area=final_town,  # type: ignore[arg-type]
        candidate_price_sgd=final_price,
        category_type=extracted_cat,
    )


def _normalize_town(town_input: Optional[str]) -> str:
    """Validates and normalizes town input against ALLOWED_PLANNING_AREAS."""
    if not town_input:
        return DEFAULT_PLANNING_AREA
    clean = town_input.strip().lower()
    for valid_area in ALLOWED_PLANNING_AREAS:
        if clean == valid_area.lower():
            return valid_area
    # Match alias
    for valid_area, pattern in _AREA_PATTERNS:
        if pattern.search(clean):
            return valid_area
    logger.warning(
        f"Town '{town_input}' not in allowed list {ALLOWED_PLANNING_AREAS}. Defaulting to {DEFAULT_PLANNING_AREA}."
    )
    return DEFAULT_PLANNING_AREA


# ---------------------------------------------------------------------------
# Main Query Decomposer Service
# ---------------------------------------------------------------------------
class QueryDecomposer:
    """
    Decomposes unstructured user pitches into structured concept attributes.
    Pulls OPEN_API_KEY (with fallback to OPENAI_API_KEY) and calls `gpt-4o-mini`.
    Provides a seamless deterministic fallback if credentials are absent or if an error occurs.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini") -> None:
        self.model = model
        # Explicit priority: passed api_key -> OPEN_API_KEY -> OPENAI_API_KEY
        self.api_key = (
            api_key
            or os.getenv("OPEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if self.api_key:
            self.api_key = self.api_key.strip()
            # Handle placeholder values in template .env
            if self.api_key.startswith("your_") or self.api_key == "":
                self.api_key = None

    def decompose(
        self,
        raw_query: str,
        override_price: Optional[float] = None,
        override_town: Optional[str] = None,
        use_mock: bool = False,
    ) -> DecomposedConcept:
        """
        Decompose pitch using gpt-4o-mini structured parsing, falling back to rule-based.
        """
        if use_mock:
            logger.info("use_mock=True: using deterministic rule-based extractor.")
            return decompose_query_rule_based(raw_query, override_price, override_town)

        if not raw_query or not raw_query.strip():
            # Return baseline concept for empty pitch
            final_town = _normalize_town(override_town)
            final_price = float(override_price) if override_price is not None else DEFAULT_PRICE_SGD
            return DecomposedConcept(
                concept_name="Generic Singapore F&B Concept",
                concept_description="A general F&B / retail offering in Singapore.",
                target_planning_area=final_town,  # type: ignore[arg-type]
                candidate_price_sgd=final_price,
                category_type="beverage",
            )

        if not self.api_key:
            logger.info("No OpenAI API key found; using deterministic rule-based extractor.")
            return decompose_query_rule_based(raw_query, override_price, override_town)

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            system_prompt = (
                "You are an expert market research analyst for Singapore micro-SMEs and consumer businesses.\n"
                "Your task is to decompose an unstructured user pitch into structured concept attributes.\n"
                "Strict extraction rules:\n"
                f"1. target_planning_area MUST strictly be one of: {ALLOWED_PLANNING_AREAS}. "
                f"If omitted, implied outside these, or ambiguous, default to '{DEFAULT_PLANNING_AREA}'.\n"
                f"2. candidate_price_sgd: The offering's unit price in Singapore Dollars (float). "
                "Meticulously scan the pitch for price indicators such as 'S$7', '$7', 'S$7.50', 'SGD 7', '7 dollars', "
                "'7 bucks', 'priced at 7', 'selling for 7', or '7/cup'. Extract the exact numeric float (e.g. 7.0). "
                f"Only if NO price is mentioned anywhere in the text should you use the default of {DEFAULT_PRICE_SGD}.\n"
                "3. category_type MUST strictly be one of: ['beverage', 'bakery', 'full_service_dining', 'retail'].\n"
                "4. concept_name: A concise, catchy name for the business (e.g. 'Artisanal Single-Origin Kopi Bar'). "
                "Do NOT include the price prefix in the concept name.\n"
                "5. concept_description: A clear operational description of the offerings."
            )

            user_prompt = f"User pitch:\n{raw_query}"

            # Use OpenAI beta parse with structured Pydantic model
            completion = client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=DecomposedConcept,
                temperature=0.1,
            )

            parsed: Optional[DecomposedConcept] = completion.choices[0].message.parsed
            if not parsed:
                raise ValueError("OpenAI returned empty structured parse.")

            # Validate target_planning_area
            if parsed.target_planning_area not in ALLOWED_PLANNING_AREAS:
                parsed.target_planning_area = _normalize_town(parsed.target_planning_area)  # type: ignore[assignment]

            # Price safeguard: If LLM returned default price (4.50) or <= 0,
            # verify whether an explicit price was stated in the raw query text
            if parsed.candidate_price_sgd == DEFAULT_PRICE_SGD or parsed.candidate_price_sgd <= 0:
                explicit_price = _extract_explicit_price(raw_query)
                if explicit_price is not None and explicit_price != parsed.candidate_price_sgd:
                    logger.info(
                        f"LLM returned default {parsed.candidate_price_sgd}, but explicit price "
                        f"S${explicit_price:.2f} was detected in raw query. Correcting candidate_price_sgd."
                    )
                    parsed.candidate_price_sgd = explicit_price

            # Apply overrides if explicitly provided
            if override_price is not None:
                parsed.candidate_price_sgd = float(override_price)
            if override_town is not None:
                parsed.target_planning_area = _normalize_town(override_town)  # type: ignore[assignment]

            return parsed

        except Exception as exc:
            logger.warning(
                f"OpenAI query decomposition encountered error ({type(exc).__name__}: {exc}). "
                "Falling back to deterministic rule-based extractor."
            )
            return decompose_query_rule_based(raw_query, override_price, override_town)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
def decompose_query(
    raw_query: str,
    override_price: Optional[float] = None,
    override_town: Optional[str] = None,
    use_mock: bool = False,
) -> DecomposedConcept:
    """
    Decomposes an unstructured user pitch into structured concept attributes.

    Parameters:
      raw_query: The user's unstructured business pitch.
      override_price: Optional price override in SGD.
      override_town: Optional planning area override.
      use_mock: If True, uses deterministic rule-based extractor without OpenAI API calls.

    Returns:
      DecomposedConcept instance with fields:
        concept_name, concept_description, target_planning_area,
        candidate_price_sgd, category_type
    """
    service = QueryDecomposer()
    return service.decompose(
        raw_query=raw_query,
        override_price=override_price,
        override_town=override_town,
        use_mock=use_mock,
    )


# Convenience alias
decompose_pitch = decompose_query


# ---------------------------------------------------------------------------
# CLI Execution Block
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimSG Query Decomposer: Extract structured concept attributes from an unstructured pitch."
    )
    parser.add_argument(
        "pitch",
        nargs="?",
        default="Artisanal Single-Origin Kopi Bar in Tampines selling pour-over kopi for S$7.00",
        help="Raw business concept pitch",
    )
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="Override candidate price in SGD",
    )
    parser.add_argument(
        "--town",
        type=str,
        default=None,
        help=f"Override planning area ({', '.join(ALLOWED_PLANNING_AREAS)})",
    )
    parser.add_argument(
        "--force-rule-based",
        action="store_true",
        help="Force deterministic rule-based extraction instead of LLM",
    )
    args = parser.parse_args()

    print(f"\n--- Decomposing Pitch ---")
    print(f"Input: {args.pitch}")
    if args.price is not None:
        print(f"Override Price: S${args.price:.2f}")
    if args.town:
        print(f"Override Town: {args.town}")

    if args.force_rule_based:
        result = decompose_query_rule_based(args.pitch, args.price, args.town)
    else:
        result = decompose_query(args.pitch, args.price, args.town)

    print("\n--- Structured Output ---")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
