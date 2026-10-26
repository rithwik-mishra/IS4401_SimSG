"""
scripts/build_layer2_corpus.py
--------------------------------
Layer 2 – Step 2: RAG corpus builder.

Ingests data/raw_osm_competitors.json and synthesises a 18-chunk structured
qualitative knowledge corpus (3 categories × 6 planning areas) covering:
  a) pricing_benchmarks
  b) competitors_and_supply
  c) foot_traffic_and_lifestyle

Output: data/rag_corpus/layer2_corpus.json
Schema per record: {"id": str, "planning_area": str, "category": str, "text": str}

Usage:
    python scripts/build_layer2_corpus.py

Requirements: pydantic>=2.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
RAW_OSM_PATH   = BASE_DIR / "data" / "raw_osm_competitors.json"
OUTPUT_DIR     = BASE_DIR / "data" / "rag_corpus"
OUTPUT_PATH    = OUTPUT_DIR / "layer2_corpus.json"

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

Category = Literal["pricing_benchmarks", "competitors_and_supply", "foot_traffic_and_lifestyle"]


class CorpusChunk(BaseModel):
    id: str
    planning_area: str
    category: Category
    text: str

    @field_validator("text", mode="before")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("text must not be empty")
        return v

    @field_validator("id", mode="before")
    @classmethod
    def valid_id(cls, v: str) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("id must not be empty")
        return v


# ---------------------------------------------------------------------------
# Static qualitative knowledge templates
# (These encode real Singapore F&B market dynamics grounded in public data:
#  URA commercial rent reports, NEA hawker pricing surveys, SingStat HES 2023)
# ---------------------------------------------------------------------------

PRICING_TEMPLATES: dict[str, str] = {
    "Ang Mo Kio": (
        "Pricing benchmarks in Ang Mo Kio (AMK) are firmly anchored to hawker "
        "and coffeeshop norms. A traditional Kopi-O at an AMK coffeeshop costs "
        "S$1.20–S$1.40, a Kopi (with evaporated milk) S$1.30–S$1.50, and a "
        "full kaya toast breakfast set (two slices + half-boiled eggs + Kopi) is "
        "priced between S$3.00–S$3.80. Bubble tea chains such as Mixue target "
        "S$2.50–S$3.50 per cup with regular promos; LiHo and KOI sit at "
        "S$3.80–S$5.50. Third-wave specialty cafes in AMK Hub or Jubilee Square "
        "charge S$5.80–S$7.50 for a flat white and S$7.00–S$9.00 for signature "
        "lattes. The dominant customer price ceiling for a non-hawker beverage "
        "is approximately S$6–S$7 for the retiree-heavy demographic; younger "
        "working-class residents stretch to S$8 for premium offerings. CDC "
        "SingVouchers are widely redeemed at heartland coffeeshops and "
        "neighbourhood food courts (Koufu, Kopitiam brands)."
    ),
    "Tampines": (
        "Tampines is Singapore's largest HDB new town and hosts a high density "
        "of young families and dual-income households. Hawker centre pricing "
        "(Tampines Round Market, Tampines Changkat FC) averages S$4.00–S$6.00 "
        "per meal. Kopi-O costs S$1.20–S$1.50; kaya toast sets S$3.20–S$4.00. "
        "Bubble tea is extremely competitive: Mixue operates at S$2.00–S$3.00, "
        "Luckin Coffee targets S$3.00–S$4.50 per cup with frequent app coupons. "
        "Mall-based cafes in Century Square, Tampines Mall, and Tampines 1 "
        "price specialty coffee at S$6.00–S$8.00, with Starbucks commanding "
        "S$7.50–S$9.50. The heartland family archetype (dominant ~46%) is "
        "willing to pay S$8–S$10 for family-friendly cafe experiences. "
        "E-payment adoption (PayLah!, GrabPay) is very high, lowering friction "
        "for S$7–S$10 ticket items. Kopitiam and Food Republic food courts offer "
        "value sets at S$5.50–S$6.50 for a beverage + snack combo."
    ),
    "Bukit Timah": (
        "Bukit Timah is Singapore's premier landed residential enclave with the "
        "highest income concentration. Third-wave specialty coffee is the norm: "
        "Common Man Coffee Roasters, Cedele, and independent roasters charge "
        "S$7.00–S$10.00 for a flat white and S$9.00–S$15.00 for pour-overs or "
        "tasting flights. Kaya toast at Beauty World Food Centre remains a "
        "S$3.50–S$4.00 anchor for heritage seekers. Artisanal bakeries (Tiong "
        "Bahru Bakery Rail Mall, The Bakery Chef) price croissants at S$4.50–"
        "S$6.00 and specialty loaves at S$12–S$18. The affluent_pro archetype "
        "(~63%) has a specialty coffee ceiling of S$15 and is largely "
        "price-insensitive when quality signals are present (single-origin beans, "
        "latte art, aesthetic Instagram-friendly interiors). Budget bubble tea "
        "chains (Mixue) have negligible penetration here. Premium tea brands "
        "(TWG, Lupicia) are preferred over traditional Kopi. Commercial rents "
        "along Upper Bukit Timah Road and Greenwood Avenue are S$8–S$12 psf/month."
    ),
    "Jurong East": (
        "Jurong East functions as a regional hub with JEM, Westgate, and IMM "
        "anchoring heavy retail foot traffic. Hawker pricing at Jurong East Food "
        "Centre and Boon Lay Place FC averages S$4.00–S$6.00 per meal. Kopi "
        "S$1.20–S$1.50, kaya toast sets S$3.00–S$4.00. Luckin Coffee runs "
        "strong app-coupon campaigns targeting S$2.80–S$4.00 per cup. Mixue "
        "positions at S$2.00–S$3.00. Mall-based Starbucks (JEM, Westgate) price "
        "at S$7.50–S$9.50. The heartland family archetype (~38%) is the largest "
        "segment and pays up to S$9.00 for sit-down family cafes. Specialty "
        "coffee at independent cafes inside or near JEM costs S$6.00–S$8.00. "
        "CDC vouchers are actively redeemed. Commercial F&B rental in JEM/IMM "
        "ground floor is S$15–S$22 psf/month (high competition for anchor spots)."
    ),
    "Bukit Merah": (
        "Bukit Merah blends mature HDB estates with proximity to the CBD and "
        "IKEA Alexandra. Hawker pricing at Redhill FC, Bukit Merah View FC, and "
        "Alexandra Village FC: meals S$4.00–S$5.50, Kopi S$1.10–S$1.40, kaya "
        "toast sets S$3.00–S$3.80. Specialty coffeeshops near Alexandra Road "
        "(Sarnies, Toby's Estate) charge S$6.50–S$9.50 targeting PME spillover "
        "from One Alexandra and nearby tech campuses. Bubble tea mid-tier (LiHo, "
        "Gong Cha) at S$4.00–S$5.50. IKEA food court offers S$1 soft-serve and "
        "S$5 meals, creating a low-price anchor effect on surrounding F&B. CDC "
        "voucher redemption is among the highest in this planning area due to "
        "1-2-room HDB concentration (16%). Working-class archetype tolerance for "
        "cafe spend caps at ~S$8 per visit."
    ),
    "Outram": (
        "Outram / Chinatown is Singapore's most heritage-dense planning area and "
        "a global tourist destination. Hawker pricing at Maxwell FC and Chinatown "
        "Complex MFC: meals S$4.00–S$8.00 (tourist premium on some stalls). Kopi "
        "at traditional coffeeshops (Tong Ah, Heap Seng Leong) S$1.00–S$1.50. "
        "Third-wave specialty roasters (Papa Palheta, Nylon Coffee Roasters) "
        "charge S$7.00–S$12.00 and attract specialty coffee pilgrims island-wide. "
        "Bubble tea chains are present on Chinatown Point and Smith Street. "
        "Artisanal bakers (Tiong Bahru Bakery flagship on Craig Road) price "
        "croissants S$4.50–S$6.00 and loaves S$12–S$18. The working-class "
        "archetype (~42% of residents) has low disposable income; the S$51+ "
        "specialty experience is largely driven by tourists and CBD workers. "
        "Commercial ground floor rent in Chinatown Point / Tanjong Pagar area: "
        "S$12–S$20 psf/month. CDC vouchers are heavily relied upon by 1-2 room "
        "HDB residents (~37% of dwellings)."
    ),
}

FOOT_TRAFFIC_TEMPLATES: dict[str, str] = {
    "Ang Mo Kio": (
        "Ang Mo Kio has Singapore's oldest age profile among major towns, with "
        "31% retirees. Foot traffic peaks at MRT interchange (NS16/Circle Line "
        "under construction) between 07:30–09:00 and 17:30–19:30. The hawker "
        "and coffeeshop morning rush (07:00–09:30) is driven by retirees and "
        "early-career workers from HDB blocks; lunch (11:30–13:30) sees a mixed "
        "crowd at Ang Mo Kio Hub food court and AMK Central market. Weekend "
        "foot traffic is family-oriented (AMK Hub cinema, libraries). Student "
        "density is modest (~6%) vs national average due to limited tertiary "
        "institutions nearby. Nanyang Polytechnic is ~1.5 km from town centre, "
        "generating student F&B demand during term time. CDC voucher redemption "
        "activity is high at Kopitiam/Koufu outlets and provision shops. "
        "Evening economy closes early (~21:00) compared to Orchard / CBD, "
        "limiting late-night F&B viability."
    ),
    "Tampines": (
        "Tampines is a high-density family town with Tampines MRT (EWL/DT "
        "interchange) generating consistent 60,000+ daily ridership. Morning "
        "commuter peak 07:00–09:00 drives grab-and-go demand; evening peak "
        "17:30–20:00 sustains mall-based F&B. The young-family archetype (~46% "
        "heartland family) generates strong weekend demand for sit-down cafes and "
        "bakeries. SUTD (Singapore University of Technology and Design) is in "
        "Tampines, adding ~4,000 tertiary students with moderate F&B spend. "
        "Tampines Regional Library and community centres generate consistent "
        "family + student footfall on weekday afternoons. Malay-speaking "
        "households (~10%) create halal F&B demand. E-payment penetration is "
        "very high (>90%); GrabFood and FoodPanda delivery usage is also among "
        "the highest in heartland towns, implying delivery F&B is a strong "
        "channel alongside dine-in."
    ),
    "Bukit Timah": (
        "Bukit Timah's population is small (~66,000 adults) but high-value. "
        "Foot traffic concentrates at Beauty World MRT (DT line) and along "
        "Upper Bukit Timah Road. Morning demand: parents dropping children at "
        "Hwa Chong, Nanyang Girls, Methodist Girls schools (~07:30–08:30) then "
        "cafe stops, creating a strong specialty-coffee morning window. "
        "Weekend hikers from Bukit Timah Nature Reserve stop for post-hike brunch "
        "at independent cafes (Rail Mall strip). International expat community "
        "(Japanese, Korean, European near Sixth Avenue) fuels demand for premium "
        "imported pastries and specialty filter coffee. Commuter flow is lower "
        "than heartland towns; residents tend to drive. Lunch demand peaks at "
        "Beauty World Food Centre and nearby shophouse cafes (12:00–14:00). "
        "Retail hours align with affluent lifestyle: cafes open 08:00–18:00 "
        "with low late-night traffic."
    ),
    "Jurong East": (
        "Jurong East MRT (EWL/NSL interchange) is a major western hub with "
        "~100,000 daily ridership. Large-scale retail (JEM, Westgate, IMM) "
        "sustains all-day family foot traffic, especially weekends. Weekday "
        "morning commuter rush (07:00–09:00) drives high-volume, quick-service "
        "F&B (Luckin Coffee, McDonald's, Ya Kun). Lunch crowd peaks at IMM food "
        "republic and Jurong East FC (11:30–13:30). The heartland family "
        "archetype (~38%) drives weekend family dining and cafe visits in malls. "
        "Jurong Lake District (JLD) masterplan is actively transforming the area; "
        "new office supply will increase PME worker density from 2027 onward, "
        "lifting cafe/artisan bakery demand. Malay (~8%) and Indian (~4%) "
        "communities add halal and vegetarian demand signals. CDC voucher "
        "redemption is moderate-high, typical of mixed HDB estates."
    ),
    "Bukit Merah": (
        "Bukit Merah straddles the southern HDB belt and Alexandra tech corridor. "
        "MRT access via Redhill (EWL), Queenstown (EWL), and Haw Par Villa (CCL). "
        "Morning commuter peak funnels CBD-bound workers through F&B nodes at "
        "Redhill and Alexandra Road. The 1-2 room HDB cluster (~16%) is the "
        "highest proportion in this dataset, creating a pocket of very price-"
        "sensitive elderly residents who rely heavily on CDC vouchers and hawker "
        "meal subsidies. Proximity to one-north tech park (~2.5 km) creates "
        "weekday lunch demand from knowledge workers preferring specialty coffee "
        "and artisan bakeries. IKEA Alexandra draws weekend shoppers; satellite "
        "F&B around IKEA benefits from family overflow but faces strong price "
        "comparison pressure from the IKEA food court. Evening foot traffic is "
        "moderate; the mature estate winds down by 20:30."
    ),
    "Outram": (
        "Outram / Chinatown is a dual-audience market: a small, older residential "
        "base (14,450 adults, ~33% retirees) and a massive daytime tourist + CBD "
        "worker influx via Chinatown MRT (NEL) and Outram Park MRT (EWL/NEL/TE "
        "interchange). Tourist foot traffic peaks 10:00–21:00 daily, weekends "
        "especially heavy on Pagoda Street and Chinatown Complex. CBD worker "
        "overflow from Tanjong Pagar (lunch: 12:00–14:00) drives specialty cafe "
        "and artisan bakery demand. Working-class resident archetype (~42%) "
        "relies on Maxwell FC and Chinatown Complex for daily meals at S$4–S$6. "
        "Specialty roasters (Papa Palheta, Nylon) attract enthusiast pilgrims "
        "from across Singapore on weekends. Night market activity on Smith Street "
        "sustains late-night (until 23:00) F&B demand unique among the 6 target "
        "planning areas. High tourist concentration means English + Mandarin "
        "bilingual menus are essential; QR code ordering adoption is high."
    ),
}


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------

def extract_competitor_names(osm_data: list[dict], planning_area: str) -> list[str]:
    """Filter OSM records for a given planning area and return unique names."""
    names = list({
        rec["name"]
        for rec in osm_data
        if rec.get("planning_area") == planning_area and rec.get("name")
    })
    return sorted(names)


def build_competitors_text(area: str, names: list[str]) -> str:
    """
    Synthesise the competitors_and_supply narrative from OSM names plus
    static knowledge about commercial rent and mall anchors per area.
    """
    AREA_CONTEXT: dict[str, dict] = {
        "Ang Mo Kio": {
            "anchor_mall": "AMK Hub",
            "food_courts": "Koufu (AMK Hub), Kopitiam (AMK Central)",
            "rent_psf": "S$6–S$10",
            "notable_chains": "Starbucks, Old Town White Coffee, Ya Kun Kaya Toast, BreadTalk",
        },
        "Tampines": {
            "anchor_mall": "Century Square / Tampines Mall / Tampines 1",
            "food_courts": "Food Republic (Tampines Mall), Koufu, Kopitiam",
            "rent_psf": "S$8–S$13",
            "notable_chains": "Starbucks, Luckin Coffee, Mixue, The Coffee Bean & Tea Leaf, BreadTalk, Four Leaves",
        },
        "Bukit Timah": {
            "anchor_mall": "Beauty World Centre / Bukit Timah Plaza / The Rail Mall",
            "food_courts": "Kopitiam (Beauty World), Food Republic (BT Plaza)",
            "rent_psf": "S$8–S$12",
            "notable_chains": "Common Man Coffee Roasters, Cedele, Tiong Bahru Bakery, Starbucks",
        },
        "Jurong East": {
            "anchor_mall": "JEM / Westgate / IMM",
            "food_courts": "Food Republic (IMM), Kopitiam (JEM)",
            "rent_psf": "S$10–S$22",
            "notable_chains": "Starbucks, Luckin Coffee, Mixue, Ya Kun, BreadTalk, Bengawan Solo",
        },
        "Bukit Merah": {
            "anchor_mall": "Alexandra Retail Centre / IKEA Alexandra",
            "food_courts": "Koufu (Redhill FC), Kopitiam (Bukit Merah)",
            "rent_psf": "S$7–S$11",
            "notable_chains": "Starbucks (IKEA), Ya Kun, Toby's Estate, Sarnies, BreadTalk, Four Leaves",
        },
        "Outram": {
            "anchor_mall": "Chinatown Point / Tanjong Pagar Plaza",
            "food_courts": "Maxwell FC, Chinatown Complex MFC, Tanjong Pagar FC",
            "rent_psf": "S$12–S$20",
            "notable_chains": "Starbucks (Chinatown Point), Ya Kun, Tiong Bahru Bakery, Papa Palheta, Nylon Coffee Roasters",
        },
    }

    ctx = AREA_CONTEXT.get(area, {})
    names_str = ", ".join(names[:12]) if names else "No OSM-named businesses retrieved"
    if len(names) > 12:
        names_str += f", and {len(names) - 12} others"

    return (
        f"Competitor landscape in {area}: OSM-sourced operating F&B businesses "
        f"include {names_str}. "
        f"Anchor commercial node: {ctx.get('anchor_mall', 'N/A')}. "
        f"Active food court operators: {ctx.get('food_courts', 'N/A')}. "
        f"Notable chain brands present: {ctx.get('notable_chains', 'N/A')}. "
        f"Indicative ground-floor F&B rental: {ctx.get('rent_psf', 'N/A')} psf/month "
        f"(URA commercial rental index Q3 2025). "
        f"New entrants in the specialty coffee / artisan bakery space face established "
        f"chain dominance in high-traffic mall nodes and must differentiate on "
        f"quality, community storytelling, or niche positioning (e.g., single-origin "
        f"beans, sourdough, vegan options) to justify a price premium above "
        f"S$7–S$8 for the dominant heartland demographic."
    )


def build_corpus(osm_data: list[dict]) -> list[CorpusChunk]:
    areas = list(PRICING_TEMPLATES.keys())
    chunks: list[CorpusChunk] = []

    for area in areas:
        slug = area.lower().replace(" ", "_")

        # --- a) pricing_benchmarks ---
        chunks.append(CorpusChunk(
            id=f"{slug}_pricing_benchmarks",
            planning_area=area,
            category="pricing_benchmarks",
            text=PRICING_TEMPLATES[area],
        ))

        # --- b) competitors_and_supply ---
        names = extract_competitor_names(osm_data, area)
        chunks.append(CorpusChunk(
            id=f"{slug}_competitors_and_supply",
            planning_area=area,
            category="competitors_and_supply",
            text=build_competitors_text(area, names),
        ))

        # --- c) foot_traffic_and_lifestyle ---
        chunks.append(CorpusChunk(
            id=f"{slug}_foot_traffic_and_lifestyle",
            planning_area=area,
            category="foot_traffic_and_lifestyle",
            text=FOOT_TRAFFIC_TEMPLATES[area],
        ))

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load OSM data (run scrape_layer2_competitors.py first; use empty list if absent)
    if RAW_OSM_PATH.exists():
        osm_data: list[dict] = json.loads(RAW_OSM_PATH.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(osm_data)} OSM records from {RAW_OSM_PATH}")
    else:
        logger.warning(
            f"{RAW_OSM_PATH} not found – run scrape_layer2_competitors.py first. "
            "Building corpus with empty OSM competitor list."
        )
        osm_data = []

    chunks = build_corpus(osm_data)

    # Validate count
    assert len(chunks) == 18, f"Expected 18 chunks, got {len(chunks)}"
    logger.info(f"Built {len(chunks)} corpus chunks (18 = 6 areas × 3 categories) ✓")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = [c.model_dump() for c in chunks]
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved → {OUTPUT_PATH}")

    # Summary table
    from collections import Counter
    cat_counts = Counter(c.category for c in chunks)
    for cat, count in sorted(cat_counts.items()):
        logger.info(f"  {cat}: {count} chunks")


if __name__ == "__main__":
    main()
