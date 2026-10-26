"""
extract_layer1_seed.py
──────────────────────
ETL script for SimSG Layer-1 Seed Data.

Pipeline:
  1. Read  data/layer1_master.csv
  2. Filter for the 5 target Singapore Planning Areas
     (Bukit Merah includes an optional "Outram" CBD alternate)
  3. Sanitise & parse all fields
  4. Normalise archetype weights so they sum to exactly 1.0000
  5. Validate every row through Pydantic v2 schemas
  6. Write the result to  data/layer1_seed.json

Usage:
  python scripts/extract_layer1_seed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator

# ─────────────────────────────────────────────────────────────
# 1.  PATHS
# ─────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "layer1_master.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "layer1_seed.json"

# ─────────────────────────────────────────────────────────────
# 2.  TARGET PLANNING AREAS
#     Bukit Merah is the primary; Outram is admitted as an
#     alternate CBD preset if it exists in the data.
# ─────────────────────────────────────────────────────────────

PRIMARY_TARGETS: List[str] = [
    "Ang Mo Kio",
    "Tampines",
    "Bukit Timah",
    "Jurong East",
    "Bukit Merah",
]

ALTERNATE_TARGETS: List[str] = [
    "Outram",  # CBD preset, included only if present in source
]

# ─────────────────────────────────────────────────────────────
# 3.  PYDANTIC v2 SCHEMAS
# ─────────────────────────────────────────────────────────────


class AgeDistribution(BaseModel):
    """Percentage share of each life-stage cohort (must sum ≤ 1.01 before norm)."""

    student_pct: float = Field(..., ge=0.0, le=1.0, description="Student / young adult share")
    early_career_pct: float = Field(..., ge=0.0, le=1.0, description="Early-career adult share")
    family_career_pct: float = Field(..., ge=0.0, le=1.0, description="Family-career adult share")
    mature_working_pct: float = Field(..., ge=0.0, le=1.0, description="Mature working adult share")
    retiree_pct: float = Field(..., ge=0.0, le=1.0, description="Retiree share")


class DwellingDistribution(BaseModel):
    """Percentage share of dwelling types (may not sum to 1 due to rounding)."""

    hdb_1_2_room_pct: float = Field(..., ge=0.0, le=1.0)
    hdb_3_room_pct: float = Field(..., ge=0.0, le=1.0)
    hdb_4_room_pct: float = Field(..., ge=0.0, le=1.0)
    hdb_5_room_exec_pct: float = Field(..., ge=0.0, le=1.0)
    condo_apt_pct: float = Field(..., ge=0.0, le=1.0)
    landed_pct: float = Field(..., ge=0.0, le=1.0)


class IncomeDistribution(BaseModel):
    """Household income bracket shares."""

    unemployed_pct: float = Field(..., ge=0.0, le=1.0)
    below3k_pct: float = Field(..., ge=0.0, le=1.0)
    income_3k_to_5999_pct: float = Field(..., ge=0.0, le=1.0)
    income_6k_to_9999_pct: float = Field(..., ge=0.0, le=1.0)
    income_10k_to_17499_pct: float = Field(..., ge=0.0, le=1.0)
    income_17500_plus_pct: float = Field(..., ge=0.0, le=1.0)


class HouseholdSizeDistribution(BaseModel):
    """Percentage of households by size band."""

    hh_size_1_to_2_pct: float = Field(..., ge=0.0, le=1.0)
    hh_size_3_to_4_pct: float = Field(..., ge=0.0, le=1.0)
    hh_size_5_plus_pct: float = Field(..., ge=0.0, le=1.0)
    mean_hh_size: float = Field(..., gt=0.0, le=10.0)


class HomeLanguageDistribution(BaseModel):
    """Primary home-language shares."""

    english_pct: float = Field(..., ge=0.0, le=1.0)
    mandarin_pct: float = Field(..., ge=0.0, le=1.0)
    chinese_dialect_pct: float = Field(..., ge=0.0, le=1.0)
    malay_pct: float = Field(..., ge=0.0, le=1.0)
    indian_pct: float = Field(..., ge=0.0, le=1.0)


class ArchetypeWeights(BaseModel):
    """
    Five SimSG persona archetype weights, normalised to sum exactly 1.0000.

    Column mapping (CSV → model field):
      weight_retiree        → retiree
      weight_student        → student
      weight_early_career   → working_class   (HDB 1-3 rm working adults)
      weight_family_career  → heartland_family (HDB 4-5 rm / Exec working adults)
      weight_affluent_pro   → affluent_pro
    """

    retiree: float = Field(..., ge=0.0, le=1.0)
    student: float = Field(..., ge=0.0, le=1.0)
    working_class: float = Field(..., ge=0.0, le=1.0, description="Maps from weight_early_career")
    heartland_family: float = Field(..., ge=0.0, le=1.0, description="Maps from weight_family_career")
    affluent_pro: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "ArchetypeWeights":
        total = round(
            self.retiree
            + self.student
            + self.working_class
            + self.heartland_family
            + self.affluent_pro,
            4,
        )
        if total != 1.0:
            raise ValueError(
                f"ArchetypeWeights sum to {total:.6f} — must be exactly 1.0000 after normalisation."
            )
        return self


class PlanningAreaRecord(BaseModel):
    """Top-level record for one Singapore Planning Area."""

    planning_area: str
    total_adult_pop_2026: int = Field(..., gt=0)
    total_households: int = Field(..., gt=0)
    age_distribution: AgeDistribution
    dwelling_distribution: DwellingDistribution
    income_distribution: IncomeDistribution
    household_size: HouseholdSizeDistribution
    home_language: HomeLanguageDistribution
    archetype_weights: ArchetypeWeights
    # Convenience flag set by the pipeline
    is_alternate_cbd_preset: Optional[bool] = False


# ─────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────


def _parse_int(value: object) -> int:
    """Convert comma-formatted strings such as '134,260' to int 134260."""
    if isinstance(value, str):
        return int(value.replace(",", "").strip())
    return int(value)


def _normalise_weights(
    retiree: float,
    student: float,
    early_career: float,
    family_career: float,
    affluent_pro: float,
) -> Dict[str, float]:
    """
    Divide each raw weight by their sum then round to 4 d.p.
    Apply a final epsilon correction to the largest weight so the
    five values always total exactly 1.0000.
    """
    raw = [retiree, student, early_career, family_career, affluent_pro]
    total_raw = sum(raw)

    if total_raw == 0:
        raise ValueError("All archetype weights are zero — cannot normalise.")

    normalised = [round(w / total_raw, 4) for w in raw]

    # Enforce exact sum by adjusting the largest weight
    deficit = round(1.0 - sum(normalised), 4)
    if deficit != 0.0:
        idx_max = normalised.index(max(normalised))
        normalised[idx_max] = round(normalised[idx_max] + deficit, 4)

    keys = ["retiree", "student", "working_class", "heartland_family", "affluent_pro"]
    return dict(zip(keys, normalised))


# ─────────────────────────────────────────────────────────────
# 5.  MAIN ETL PIPELINE
# ─────────────────────────────────────────────────────────────


def extract_layer1_seed() -> List[PlanningAreaRecord]:
    """
    Full ETL pipeline.
    Returns a list of validated PlanningAreaRecord objects.
    """

    # ── 5a. Read CSV ──────────────────────────────────────────
    if not CSV_PATH.exists():
        sys.exit(f"[ERROR] Source file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # ── 5b. Filter target planning areas ─────────────────────
    all_targets = PRIMARY_TARGETS + [
        t for t in ALTERNATE_TARGETS if t in df["planning_area"].values
    ]
    df = df[df["planning_area"].isin(all_targets)].copy()

    if df.empty:
        sys.exit("[ERROR] None of the target planning areas were found in the CSV.")

    missing = set(PRIMARY_TARGETS) - set(df["planning_area"].values)
    if missing:
        sys.exit(f"[ERROR] Missing primary planning areas in source data: {missing}")

    # ── 5c. Parse integer columns ─────────────────────────────
    df["total_adult_pop_2026"] = df["total_adult_pop_2026"].apply(_parse_int)
    df["total_households"] = df["total_households"].apply(_parse_int)

    # ── 5d. Build validated records ───────────────────────────
    records: List[PlanningAreaRecord] = []

    for _, row in df.iterrows():
        area = row["planning_area"]

        # Normalise archetype weights
        normalised = _normalise_weights(
            retiree=float(row["weight_retiree"]),
            student=float(row["weight_student"]),
            early_career=float(row["weight_early_career"]),
            family_career=float(row["weight_family_career"]),
            affluent_pro=float(row["weight_affluent_pro"]),
        )

        record = PlanningAreaRecord(
            planning_area=area,
            total_adult_pop_2026=int(row["total_adult_pop_2026"]),
            total_households=int(row["total_households"]),
            is_alternate_cbd_preset=(area in ALTERNATE_TARGETS),
            age_distribution=AgeDistribution(
                student_pct=float(row["student_pct"]),
                early_career_pct=float(row["early_career_pct"]),
                family_career_pct=float(row["family_career_pct"]),
                mature_working_pct=float(row["mature_working_pct"]),
                retiree_pct=float(row["retiree_pct"]),
            ),
            dwelling_distribution=DwellingDistribution(
                hdb_1_2_room_pct=float(row["dwell_hdb_1_2_room_pct"]),
                hdb_3_room_pct=float(row["dwell_hdb_3_room_pct"]),
                hdb_4_room_pct=float(row["dwell_hdb_4_room_pct"]),
                hdb_5_room_exec_pct=float(row["dwell_hdb_5_room_exec_pct"]),
                condo_apt_pct=float(row["dwell_condo_apt_pct"]),
                landed_pct=float(row["dwell_landed_pct"]),
            ),
            income_distribution=IncomeDistribution(
                unemployed_pct=float(row["income_unemployed_pct"]),
                below3k_pct=float(row["income_below3k_pct"]),
                income_3k_to_5999_pct=float(row["income_3k_to_5999_pct"]),
                income_6k_to_9999_pct=float(row["income_6k_to_9999_pct"]),
                income_10k_to_17499_pct=float(row["income_10k_to_17499_pct"]),
                income_17500_plus_pct=float(row["income_17500_plus_pct"]),
            ),
            household_size=HouseholdSizeDistribution(
                hh_size_1_to_2_pct=float(row["hh_size_1_to_2_pct"]),
                hh_size_3_to_4_pct=float(row["hh_size_3_to_4_pct"]),
                hh_size_5_plus_pct=float(row["hh_size_5_plus_pct"]),
                mean_hh_size=float(row["mean_hh_size"]),
            ),
            home_language=HomeLanguageDistribution(
                english_pct=float(row["english_pct"]),
                mandarin_pct=float(row["mandarin_pct"]),
                chinese_dialect_pct=float(row["chinese_dialect_pct"]),
                malay_pct=float(row["malay_pct"]),
                indian_pct=float(row["indian_pct"]),
            ),
            archetype_weights=ArchetypeWeights(**normalised),
        )

        records.append(record)

    return records


# ─────────────────────────────────────────────────────────────
# 6.  SERIALISE & WRITE JSON
# ─────────────────────────────────────────────────────────────


def _serialise(records: List[PlanningAreaRecord]) -> List[dict]:
    """Convert Pydantic models to plain dicts (JSON-serialisable)."""
    return [r.model_dump() for r in records]


def write_json(records: List[PlanningAreaRecord]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialise(records)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n[OK] Wrote {len(payload)} records -> {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────
# 7.  CLI EXECUTION CHECK
# ─────────────────────────────────────────────────────────────

SEPARATOR = "-" * 68


def print_weight_summary(records: List[PlanningAreaRecord]) -> None:
    """
    Print a formatted table of normalised archetype weights for all towns
    and confirm each row sums to exactly 1.0000.
    """
    archetypes = ["retiree", "student", "working_class", "heartland_family", "affluent_pro"]
    col_w = 17  # fixed column width

    header_area = f"{'Planning Area':<22}"
    header_cols = "".join(f"{a:>{col_w}}" for a in archetypes)
    header_sum  = f"{'SUM':>{col_w}}"

    print(f"\n{SEPARATOR}")
    print("  SimSG Layer-1 Seed - Normalised Archetype Weights")
    print(SEPARATOR)
    print(f"{header_area}{header_cols}{header_sum}")
    print(SEPARATOR)

    all_ok = True
    for rec in records:
        w = rec.archetype_weights
        vals = [w.retiree, w.student, w.working_class, w.heartland_family, w.affluent_pro]
        total = round(sum(vals), 4)

        tag = "[OK]" if total == 1.0 else "[FAIL]"
        if total != 1.0:
            all_ok = False

        row_vals = "".join(f"{v:>{col_w}.4f}" for v in vals)
        row_sum  = f"{total:>{col_w}.4f}  {tag}"
        alt_flag = "  [CBD alt]" if rec.is_alternate_cbd_preset else ""
        print(f"{rec.planning_area + alt_flag:<22}{row_vals}{row_sum}")

    print(SEPARATOR)
    if all_ok:
        print("  [PASS] All archetype weight rows sum to exactly 1.0000.")
    else:
        print("  [FAIL] One or more rows FAILED the 1.0000 sum check!")
    print(SEPARATOR)


# ─────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'=' * 68}")
    print("  SimSG - extract_layer1_seed  ETL Pipeline")
    print(f"{'=' * 68}")
    print(f"  Source : {CSV_PATH}")
    print(f"  Output : {OUTPUT_PATH}")

    print("\n[1/3] Extracting & validating records ...")
    records = extract_layer1_seed()
    print(f"      [OK] {len(records)} planning areas validated.")

    print("\n[2/3] Writing layer1_seed.json ...")
    write_json(records)

    print("\n[3/3] Weight normalisation check:")
    print_weight_summary(records)
