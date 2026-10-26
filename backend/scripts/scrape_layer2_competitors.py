"""
scripts/scrape_layer2_competitors.py
-------------------------------------
Layer 2 – Step 1: OSM Overpass API scraper.

Queries operating food & beverage establishments within a 1,200 m radius of
each target Singapore town centre, deduplicates results, sanitises entity
names, and writes the raw list to data/raw_osm_competitors.json.

Usage:
    python scripts/scrape_layer2_competitors.py

Requirements: requests, pydantic>=2.0
"""

from __future__ import annotations

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional

import requests
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 1200
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw_osm_competitors.json"

TOWN_CENTRES: dict[str, tuple[float, float]] = {
    "Ang Mo Kio":  (1.3691, 103.8454),
    "Tampines":    (1.3526, 103.9446),
    "Bukit Timah": (1.3294, 103.8021),
    "Jurong East": (1.3329, 103.7436),
    "Bukit Merah": (1.2819, 103.8239),
    "Outram":      (1.2803, 103.8406),
}

# OSM tags to match (amenity OR shop filters)
OSM_FILTERS = [
    ('amenity', 'cafe'),
    ('amenity', 'fast_food'),
    ('amenity', 'food_court'),
    ('shop',    'bakery'),
    ('shop',    'coffee'),
    ('shop',    'tea'),
    ('shop',    'pastry'),
]

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class Competitor(BaseModel):
    osm_id: int
    osm_type: str                     # "node" | "way" | "relation"
    name: str
    amenity: Optional[str] = None
    shop: Optional[str] = None
    planning_area: str
    lat: Optional[float] = None
    lon: Optional[float] = None

    @field_validator("name", mode="before")
    @classmethod
    def sanitise_name(cls, v: str) -> str:
        """Strip control characters, normalise whitespace, title-case if ALL-CAPS."""
        v = re.sub(r"[\x00-\x1F\x7F]", "", str(v)).strip()
        v = re.sub(r"\s{2,}", " ", v)
        if v.isupper() and len(v) > 2:
            v = v.title()
        return v


# ---------------------------------------------------------------------------
# Overpass query builder
# ---------------------------------------------------------------------------

def build_overpass_query(lat: float, lon: float, radius: int) -> str:
    """Return an Overpass QL query string for the given centre + radius."""
    union_parts: list[str] = []
    for tag_key, tag_val in OSM_FILTERS:
        for element in ("node", "way", "relation"):
            union_parts.append(
                f'  {element}["{tag_key}"="{tag_val}"](around:{radius},{lat},{lon});'
            )
    inner = "\n".join(union_parts)
    return f"[out:json][timeout:60];\n(\n{inner}\n);\nout center;"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def fetch_competitors(town: str, lat: float, lon: float) -> list[Competitor]:
    """Query Overpass API and parse results for a single town centre."""
    query = build_overpass_query(lat, lon, RADIUS_M)
    logger.info(f"Querying OSM for '{town}' ({lat}, {lon}) …")

    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=90,
            headers={"User-Agent": "SimSG-Layer2-Scraper/1.0 (academic project)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Overpass request failed for '{town}': {exc}")
        return []

    elements: list[dict] = response.json().get("elements", [])
    logger.info(f"  → {len(elements)} raw elements returned")

    competitors: list[Competitor] = []
    for el in elements:
        tags = el.get("tags", {})
        raw_name = tags.get("name") or tags.get("name:en") or tags.get("brand")
        if not raw_name:
            continue  # skip unnamed establishments

        # Resolve centre coordinates for ways/relations
        if el["type"] == "node":
            lat_c, lon_c = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat_c, lon_c = center.get("lat"), center.get("lon")

        try:
            comp = Competitor(
                osm_id=el["id"],
                osm_type=el["type"],
                name=raw_name,
                amenity=tags.get("amenity"),
                shop=tags.get("shop"),
                planning_area=town,
                lat=lat_c,
                lon=lon_c,
            )
            competitors.append(comp)
        except Exception as exc:
            logger.debug(f"Skipping element {el.get('id')}: {exc}")

    return competitors


def deduplicate(competitors: list[Competitor]) -> list[Competitor]:
    """
    Deduplicate by (planning_area, normalised_name).
    Within a planning area two entries with identical clean names are collapsed;
    across areas the same chain brand may legitimately appear multiple times.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Competitor] = []
    for c in competitors:
        key = (c.planning_area, re.sub(r"\W+", "", c.name.lower()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Fallback: synthetic seed data (used when Overpass is unreachable)
# ---------------------------------------------------------------------------

SYNTHETIC_FALLBACK: dict[str, list[dict]] = {
    "Ang Mo Kio": [
        {"name": "Ang Mo Kio Hawker Centre", "amenity": "food_court"},
        {"name": "Ya Kun Kaya Toast (AMK Hub)", "amenity": "cafe"},
        {"name": "Koufu AMK Hub", "amenity": "food_court"},
        {"name": "Starbucks AMK Hub", "amenity": "cafe"},
        {"name": "Mixue (AMK)", "amenity": "cafe"},
        {"name": "Old Town White Coffee", "amenity": "cafe"},
        {"name": "Paris Baguette AMK", "shop": "bakery"},
        {"name": "BreadTalk AMK Hub", "shop": "bakery"},
    ],
    "Tampines": [
        {"name": "Tampines Round Market & Food Centre", "amenity": "food_court"},
        {"name": "Tampines Mall Food Republic", "amenity": "food_court"},
        {"name": "Starbucks Tampines", "amenity": "cafe"},
        {"name": "Luckin Coffee Tampines", "amenity": "cafe"},
        {"name": "Mixue Tampines", "amenity": "cafe"},
        {"name": "The Coffee Bean & Tea Leaf Tampines", "amenity": "cafe"},
        {"name": "BreadTalk Tampines", "shop": "bakery"},
        {"name": "Four Leaves Bakery Tampines", "shop": "bakery"},
    ],
    "Bukit Timah": [
        {"name": "Beauty World Food Centre", "amenity": "food_court"},
        {"name": "Sora Boru Cafe", "amenity": "cafe"},
        {"name": "Common Man Coffee Roasters (Bukit Timah)", "amenity": "cafe"},
        {"name": "The Bakery Chef", "shop": "bakery"},
        {"name": "Cedele Bukit Timah", "amenity": "cafe"},
        {"name": "Starbucks Bukit Timah Plaza", "amenity": "cafe"},
        {"name": "Jewel Cafe", "amenity": "cafe"},
        {"name": "Tiong Bahru Bakery (Rail Mall)", "shop": "bakery"},
    ],
    "Jurong East": [
        {"name": "Jurong East Food Centre", "amenity": "food_court"},
        {"name": "IMM Food Republic", "amenity": "food_court"},
        {"name": "Starbucks JEM", "amenity": "cafe"},
        {"name": "Luckin Coffee Jurong East", "amenity": "cafe"},
        {"name": "Mixue Westgate", "amenity": "cafe"},
        {"name": "Ya Kun Kaya Toast (JEM)", "amenity": "cafe"},
        {"name": "BreadTalk JEM", "shop": "bakery"},
        {"name": "Bengawan Solo JEM", "shop": "bakery"},
    ],
    "Bukit Merah": [
        {"name": "Redhill Food Centre", "amenity": "food_court"},
        {"name": "Bukit Merah View Food Centre", "amenity": "food_court"},
        {"name": "Starbucks IKEA Alexandria", "amenity": "cafe"},
        {"name": "Ya Kun Kaya Toast (Alexandra)", "amenity": "cafe"},
        {"name": "Sarnies Cafe", "amenity": "cafe"},
        {"name": "BreadTalk Alexandra", "shop": "bakery"},
        {"name": "Four Leaves Bukit Merah", "shop": "bakery"},
        {"name": "Toby's Estate", "amenity": "cafe"},
    ],
    "Outram": [
        {"name": "Chinatown Complex Market & Food Centre", "amenity": "food_court"},
        {"name": "Maxwell Food Centre", "amenity": "food_court"},
        {"name": "Tong Ah Eating House", "amenity": "cafe"},
        {"name": "Starbucks Chinatown Point", "amenity": "cafe"},
        {"name": "Ya Kun Kaya Toast (Chinatown)", "amenity": "cafe"},
        {"name": "Tiong Bahru Bakery Flagship", "shop": "bakery"},
        {"name": "Papa Palheta", "amenity": "cafe"},
        {"name": "Nylon Coffee Roasters", "amenity": "cafe"},
    ],
}


def build_fallback_competitors() -> list[Competitor]:
    """Create synthetic Competitor records from the hardcoded fallback seed."""
    results: list[Competitor] = []
    fake_id = 9_000_000_000
    for town, entries in SYNTHETIC_FALLBACK.items():
        lat, lon = TOWN_CENTRES[town]
        for entry in entries:
            comp = Competitor(
                osm_id=fake_id,
                osm_type="node",
                name=entry["name"],
                amenity=entry.get("amenity"),
                shop=entry.get("shop"),
                planning_area=town,
                lat=lat,
                lon=lon,
            )
            results.append(comp)
            fake_id += 1
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_competitors: list[Competitor] = []
    overpass_ok = False

    for town, (lat, lon) in TOWN_CENTRES.items():
        results = fetch_competitors(town, lat, lon)
        if results:
            overpass_ok = True
        all_competitors.extend(results)
        time.sleep(2)  # polite rate-limit

    if not overpass_ok:
        logger.warning(
            "Overpass returned no named results (network issue or rate-limit). "
            "Falling back to synthetic seed data."
        )
        all_competitors = build_fallback_competitors()

    deduped = deduplicate(all_competitors)
    logger.info(f"Total unique competitors after dedup: {len(deduped)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.model_dump() for c in deduped]
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
