"""
SimSG - draw N synthetic residents (personas) for one planning area.

Run after clean_data.py and fit.py. Try it:
    python sampler.py "Ang Mo Kio" 10            # 10 personas, random seed
    python sampler.py "Ang Mo Kio" 10 --seed 42  # same 10 personas every time
    python sampler.py --check "Ang Mo Kio"       # draw 20,000 and compare with the source tables

Or from other code:
    from sampler import sample_personas, list_areas
    panel, seed = sample_personas("Tampines", 30)

Order of draws (each step uses only the tables named):
    1. dwelling + household income band   fitted table (area)                 fit.py output
    2. sex + age group (adults only)      resident area x age x sex            independent of step 1
    3. ethnic group                       resident area x sex x ethnic group   given area and sex
    4. spending per household member      HES 2023 by dwelling                 looked up, not drawn
"""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "sheets"
PROC = DATA / "processed"

MIN_AGE = 20   # personas are adults aged 20 and over

SPEND_ROWS = {
    "Food and Beverage Serving Services": "spend_eating_out",
    "Food and Non-Alcoholic Beverages": "spend_groceries",
    "Total": "spend_total",
}

DWELLING_TEXT = {
    "1-2 room": "HDB 1- or 2-room flat",
    "3 room": "HDB 3-room flat",
    "4 room": "HDB 4-room flat",
    "5 room/exec": "HDB 5-room or executive flat",
    "condo": "condominium or private apartment",
    "landed": "landed property",
}


@lru_cache(maxsize=1)
def _tables():
    fitted = pd.read_csv(PROC / "fitted_area_dwelling_income.csv")
    age = pd.read_csv(PROC / "resident_area_age_sex.csv")
    age = age[age["age_lo"] >= MIN_AGE]
    eth = pd.read_csv(PROC / "resident_area_sex_ethnic.csv")
    spend = pd.read_csv(PROC / "spend_per_member_dwelling.csv")
    spend = (spend[spend["category"].isin(SPEND_ROWS)]
             .pivot(index="dwelling", columns="category", values="spend_per_member")
             .rename(columns=SPEND_ROWS))
    return fitted, age, eth, spend


def list_areas():
    return sorted(_tables()[0]["area"].unique())


def _draw(df, weight_col, n, rng):
    p = df[weight_col].to_numpy(float)
    idx = rng.choice(len(df), size=n, p=p / p.sum())
    return df.iloc[idx].reset_index(drop=True)


def income_text(lo, hi):
    if pd.isna(hi):
        return f"${int(lo):,} or more"
    if lo == 0:
        return f"below ${int(hi) + 1:,}"
    return f"${int(lo):,} to ${int(hi):,}"


def sample_personas(area, n, seed=None):
    """Return (DataFrame of n personas, seed used). Same area, n and seed give the same panel."""
    fitted, age, eth, spend = _tables()
    if area not in set(fitted["area"]):
        raise ValueError(f"No household data for {area!r}. Choose one of: {', '.join(list_areas())}")
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 1_000_000))
    # One independent random stream per step, so persona 1..k stay the same when n grows
    rng_hh, rng_age, rng_eth = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))

    # 1. dwelling + income band, weighted by residents (households x average household size)
    hh = _draw(fitted[fitted["area"] == area], "prob_in_area", n, rng_hh)

    # 2. sex + age group, adults only
    people = _draw(age[age["area"] == area], "residents", n, rng_age)

    # 3. ethnic group given area and sex
    e = eth[eth["area"] == area]
    ethnic = []
    for sex in people["sex"]:
        options = e[e["sex"] == sex]
        ethnic.append(_draw(options, "residents", 1, rng_eth)["ethnic_group"].iloc[0])

    panel = pd.DataFrame({
        "persona_id": [f"{area[:3].upper()}-{seed}-{i + 1:03d}" for i in range(n)],
        "area": area,
        "sex": people["sex"],
        "age_group": people["age_group"],
        "ethnic_group": ethnic,
        "dwelling": hh["dwelling"],
        "dwelling_text": hh["dwelling"].map(DWELLING_TEXT),
        "household_income_band": hh["income_band"],
        "household_income_text": [income_text(lo, hi) for lo, hi in zip(hh["income_lo"], hh["income_hi"])],
        "income_lo": hh["income_lo"],
    })
    # 4. spending lookup (S$ per household member per month, HES 2023 national average for the dwelling type)
    panel = panel.join(spend, on="dwelling")
    panel["seed"] = seed
    return panel, seed


def check(area, n=20000):
    """Draw a large panel and compare its shares with the source tables."""
    fitted, age, eth, _ = _tables()
    panel, seed = sample_personas(area, n, seed=1)

    def compare(name, drawn, expected):
        df = pd.DataFrame({"drawn": drawn, "expected": expected}).fillna(0)
        gap = (df["drawn"] - df["expected"]).abs().max()
        print(f"\n{name}: largest gap {gap:.3f}")
        print(df.round(3).to_string())

    f = fitted[fitted["area"] == area]
    compare("Dwelling share", panel["dwelling"].value_counts(normalize=True),
            f.groupby("dwelling")["prob_in_area"].sum())
    a = age[age["area"] == area]
    compare("Sex share (adults)", panel["sex"].value_counts(normalize=True),
            a.groupby("sex")["residents"].sum() / a["residents"].sum())
    compare("Age share (adults)", panel["age_group"].value_counts(normalize=True),
            a.groupby("age_group")["residents"].sum() / a["residents"].sum())
    e = eth[eth["area"] == area]
    compare("Ethnic share (all ages)", panel["ethnic_group"].value_counts(normalize=True),
            e.groupby("ethnic_group")["residents"].sum() / e["residents"].sum())
    again, _ = sample_personas(area, 50, seed=7)
    same, _ = sample_personas(area, 50, seed=7)
    print("\nSame seed gives the same panel:", again.equals(same))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("area", nargs="?", default="Ang Mo Kio")
    ap.add_argument("n", nargs="?", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check(args.area)
        return
    panel, seed = sample_personas(args.area, args.n, args.seed)
    pd.set_option("display.width", 200)
    print(f"Seed: {seed}")
    print(panel[["persona_id", "sex", "age_group", "ethnic_group", "dwelling",
                 "household_income_text", "spend_eating_out", "spend_total"]].to_string(index=False))


if __name__ == "__main__":
    main()