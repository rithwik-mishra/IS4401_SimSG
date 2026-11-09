"""
E1 - Persona sampler fidelity: do the synthetic personas look like the census?

Idea: the census says what share of a town's residents is in each group (e.g. 35% live in 3-room
flats). Draw many personas, count them the same way, and see how close the two lists of shares are.

Checks (each one prints a summary and saves a CSV in data/evaluation/e1/):
  1.  fit.py's table adds up to each town's published totals
  1b. all towns added up keep the national income pattern within each home type
  2.  personas vs census, every town
  4.  how far a normal 30-persona panel is from the census just by chance
  5.  seniors: labour force (60+) and living arrangement (65+) vs the national tables
  6.  step 14: retired seniors living alone with a working-household income, before vs after
  7.  same seed -> same personas

Run from backend/:
    python scripts/evaluate_sampler.py               # 40,000 personas per town
    python scripts/evaluate_sampler.py --n 2000      # quicker
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))  # so "app" can be imported when running from scripts/

from app.services import sampler as new_sampler  # noqa: E402

OUT = BACKEND / "data" / "evaluation" / "e1"
TABLES = BACKEND / "data" / "sheets" / "processed"

# Six towns for the slower checks (4 and 7): a mix of mature, new, central and private-housing towns.
CHECK_TOWNS = ["Ang Mo Kio", "Bukit Merah", "Bukit Timah", "Jurong East", "Outram", "Tampines"]
HOME_TYPES = ["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]
SMALL_FLATS = HOME_TYPES[:2]   # 1-2 room, 3 room
BIGGER_HOMES = HOME_TYPES[2:]  # 4 room and up

# Coarser groups for the 30-persona panel check (check 4): a small panel can't fill 19 income bands.
# The income edges match census band edges, so no census band is split.
AGE_GROUPS = [(20, 24, "20-24"), (25, 34, "25-34"), (35, 49, "35-49"), (50, 64, "50-64"), (65, 200, "65+")]
INCOME_GROUPS = [(0, "below S$1k"), (1000, "S$1k-2,999"), (3000, "S$3k-5,999"), (6000, "S$6k-9,999"),
                 (10000, "S$10k-17,499"), (17500, "S$17.5k+")]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def to_shares(counts):
    """Counts -> shares that add up to 1. {"a": 3, "b": 1} -> {"a": 0.75, "b": 0.25}"""
    total = sum(counts.values())
    if not total:
        return {}
    return {k: v / total for k, v in counts.items()}


def compare(personas, census):
    """
    Compare two lists of shares. Returns:
      gap      biggest single difference, in percentage points (+ = too many personas)
      category the group with that biggest difference
      tvd      share of personas in the "wrong" group (half the sum of all differences)
      srmse    typical difference relative to the average group size
    A group missing on one side counts as 0%.
    """
    groups = sorted(set(personas) | set(census))
    diffs = np.array([personas.get(g, 0.0) - census.get(g, 0.0) for g in groups])
    census_shares = np.array([census.get(g, 0.0) for g in groups])
    worst = int(np.abs(diffs).argmax())
    gap = 100 * float(diffs[worst])
    tvd = 0.5 * float(np.abs(diffs).sum())
    srmse = float(np.sqrt((diffs ** 2).mean()) / census_shares.mean())
    return gap, groups[worst], tvd, srmse


def age_group(age):
    for low, high, label in AGE_GROUPS:
        if low <= age <= high:
            return label


def income_group(income):
    label = None
    for low, name in INCOME_GROUPS:
        if income >= low:
            label = name
    return label


def read_csv(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_row(extra, gap, category, tvd, srmse, gap_decimals=2):
    """One CSV row: the labels in `extra`, then the four scores from compare()."""
    return {**extra, "max_gap_pts": round(gap, gap_decimals), "category": category,
            "tvd": round(tvd, 4), "srmse": round(srmse, 4)}


# ---------------------------------------------------------------------------
# What a town should look like (census) and what the personas look like
# ---------------------------------------------------------------------------

def census_shares(town):
    """The census shares for one town, in the six ways we count people."""
    households, people, _ = new_sampler._load_census_tables()  # the sampler's own data
    home, income, income_coarse = Counter(), Counter(), Counter()
    for row in households[town]:  # home type x income rows, weighted by residents
        home[row["dwelling"]] += row["weight"]
        income[row["income_label"]] += row["weight"]
        income_coarse[income_group(row["income_lo"])] += row["weight"]
    age_sex, age_coarse, sex = Counter(), Counter(), Counter()
    for row in people[town]:  # age group x sex rows, adults 20+
        age_sex[f"{row['age_group']} {row['sex']}"] += row["weight"]
        age_coarse[age_group(row["age_lo"])] += row["weight"]
        sex[row["sex"]] += row["weight"]
    return {"dwelling": to_shares(home), "income_band": to_shares(income), "income_coarse": to_shares(income_coarse),
            "age_sex": to_shares(age_sex), "age_coarse": to_shares(age_coarse), "sex": to_shares(sex)}


def persona_shares(personas):
    """The personas' shares, counted the same six ways as census_shares()."""
    home, income, income_coarse, age_sex, age_coarse, sex = (Counter() for _ in range(6))
    for p in personas:
        home[p.dwelling_type] += 1
        income[p.income_bracket] += 1
        income_coarse[income_group(p.monthly_household_income_sgd)] += 1
        age_sex[f"{p.age_bracket} {p.gender}"] += 1
        age_coarse[age_group(p.age)] += 1
        sex[p.gender] += 1
    return {"dwelling": to_shares(home), "income_band": to_shares(income), "income_coarse": to_shares(income_coarse),
            "age_sex": to_shares(age_sex), "age_coarse": to_shares(age_coarse), "sex": to_shares(sex)}


# ---------------------------------------------------------------------------
# Check 1: fit.py vs published town totals
# ---------------------------------------------------------------------------

def check_census_fit():
    # Add up households per (town, home type) and per (town, income band)
    fitted_home, fitted_income = defaultdict(float), defaultdict(float)
    for row in read_csv("fitted_area_dwelling_income.csv"):
        fitted_home[(row["area"], row["dwelling"])] += float(row["households_fitted"])
        fitted_income[(row["area"], row["income_band"])] += float(row["households_fitted"])
    published_home, published_income = defaultdict(float), defaultdict(float)
    for row in read_csv("household_area_dwelling.csv"):
        published_home[(row["area"], row["dwelling"])] += float(row["households"])
    for row in read_csv("household_area_income.csv"):
        published_income[(row["area"], row["income_band"])] += float(row["households"])

    rows = []
    for town in new_sampler.list_supported_areas():
        for table, fitted, published in (("dwelling", fitted_home, published_home),
                                         ("income_band", fitted_income, published_income)):
            fitted_town = to_shares({group: v for (t, group), v in fitted.items() if t == town})
            published_town = to_shares({group: v for (t, group), v in published.items() if t == town})
            score = compare(fitted_town, published_town)
            rows.append(result_row({"area": town, "table": table}, *score, gap_decimals=3))
    return rows


# ---------------------------------------------------------------------------
# Check 1b: all towns added up vs the national income x home type table
# (fit.py starts from this national table, so this is a consistency check)
# ---------------------------------------------------------------------------

def check_joint_vs_national():
    fitted, national = defaultdict(float), defaultdict(float)
    for row in read_csv("fitted_area_dwelling_income.csv"):
        fitted[(row["dwelling"], row["income_band"])] += float(row["households_fitted"])
    for row in read_csv("household_income_dwelling.csv"):
        national[(row["dwelling"], row["income_band"])] += float(row["households"])

    rows = []
    for home in HOME_TYPES:
        fitted_mix = to_shares({band: v for (h, band), v in fitted.items() if h == home})
        national_mix = to_shares({band: v for (h, band), v in national.items() if h == home})
        score = compare(fitted_mix, national_mix)
        rows.append(result_row({"area": "all towns", "table": f"income within {home}"}, *score))
    return rows


# ---------------------------------------------------------------------------
# Check 2: personas vs census
# ---------------------------------------------------------------------------

def check_persona_mix(sampler_name, draw, towns, n, seed, tables):
    rows = []
    for town in towns:
        personas = persona_shares(draw(town, n, seed))
        census = census_shares(town)
        for table in tables:
            score = compare(personas[table], census[table])
            rows.append(result_row({"sampler": sampler_name, "area": town, "table": table}, *score))
    return rows


# ---------------------------------------------------------------------------
# Check 4: one 30-persona panel vs census, many times (chance alone)
# ---------------------------------------------------------------------------

def check_panel_noise(towns, panels, size=30):
    rows = []
    for town in towns:
        census = census_shares(town)
        tvds = {"dwelling": [], "income_coarse": [], "age_coarse": []}
        for seed in range(panels):
            panel = persona_shares(new_sampler.sample_synthetic_cohort(town, size, seed=seed))
            for table in tvds:
                tvds[table].append(compare(panel[table], census[table])[2])
        for table, values in tvds.items():
            rows.append({"area": town, "table": table, "panel_size": size, "panels": panels,
                         "mean_tvd": round(float(np.mean(values)), 3),
                         "p90_tvd": round(float(np.percentile(values, 90)), 3)})
    return rows


# ---------------------------------------------------------------------------
# Check 5: seniors vs the national 2025 tables
# ---------------------------------------------------------------------------

def check_seniors(personas):
    national_rate, national_living, _ = new_sampler._load_senior_tables()
    rows = []

    # Labour force: in each age group x sex, what share of personas aged 60+ is in the labour force?
    in_lf, total = Counter(), Counter()
    for p in personas:
        if p.in_labour_force is not None:
            group = (new_sampler._labour_force_age_group(p.age, p.age_bracket), p.gender)
            in_lf[group] += p.in_labour_force
            total[group] += 1
    for group in sorted(total):
        mine, national = in_lf[group] / total[group], national_rate[group]
        rows.append({"check": "labour_force_rate", "group": f"{group[0]} {group[1]}", "personas": total[group],
                     "synthetic": round(mine, 4), "national": round(national, 4),
                     "gap_pts": round(100 * (mine - national), 2)})

    # Living arrangement: in each age group, what share of personas aged 65+ has each arrangement?
    counts = defaultdict(Counter)
    for p in personas:
        if p.living_arrangement is not None:
            counts[new_sampler._living_age_group(p.age, p.age_bracket)][p.living_arrangement] += 1
    for group in sorted(counts):
        mine = to_shares(counts[group])
        national = to_shares(dict(national_living[group]))
        for arrangement in sorted(set(national) | set(mine)):
            a, b = mine.get(arrangement, 0.0), national.get(arrangement, 0.0)
            rows.append({"check": "living_arrangement", "group": f"{group} {arrangement}",
                         "personas": sum(counts[group].values()),
                         "synthetic": round(a, 4), "national": round(b, 4), "gap_pts": round(100 * (a - b), 2)})
    return rows


# ---------------------------------------------------------------------------
# Check 6: step 14 vs drawing the home without looking at the person
# ---------------------------------------------------------------------------

def check_household_plausibility(towns, n, seed):
    """
    Same people, two ways of picking their home: step 14 (the real sampler) and an independent
    draw from the town's census rows (how it worked before step 14). Towns are added up with
    weights = town adult population, so the total looks like Singapore and can be compared with
    the national mix of 1-person and 3+-person households.
    """
    _, people, _ = new_sampler._load_census_tables()

    # National reference: which home types 1-person and 3+-person households live in
    one_person, three_plus = Counter(), Counter()
    for row in read_csv("household_dwelling_size.csv"):
        size = int(row["household_size"])
        if size == 1:
            one_person[row["dwelling"]] += float(row["households"])
        elif size >= 3:
            three_plus[row["dwelling"]] += float(row["households"])
    ref_alone, ref_family = to_shares(one_person), to_shares(three_plus)

    methods = ("step 14 (person first)", "independent home draw")
    retired_alone = {m: 0.0 for m in methods}         # retired seniors living alone (weighted)
    retired_alone_earning = {m: 0.0 for m in methods}  # ...of whom household income >= S$1,000
    alone_homes = {m: Counter() for m in methods}      # home types of seniors living alone
    family_homes = {m: Counter() for m in methods}     # home types of seniors with spouse and child

    rng = np.random.default_rng(seed + 1)
    for i, town in enumerate(towns):
        weight = sum(r["weight"] for r in people[town]) / n
        plan = new_sampler._household_plan(town)
        for p in new_sampler.sample_synthetic_cohort(town, n, seed=seed + i):  # own seed per town
            group = new_sampler._household_group(p.age, p.in_labour_force, p.living_arrangement)
            other = plan["rows"][new_sampler._pick(plan["cdf"]["rest"], rng.random())]  # independent draw
            for method, home, income in ((methods[0], p.dwelling_type, p.monthly_household_income_sgd),
                                         (methods[1], other["dwelling"], other["income_lo"])):
                if group == "alone_retired":
                    retired_alone[method] += weight
                    retired_alone_earning[method] += weight * (income >= 1000)
                if group in ("alone_retired", "alone_working"):
                    alone_homes[method][home] += weight
                if group == "with_family":
                    family_homes[method][home] += weight

    def pct_in(shares, homes, decimals=1):
        return round(100 * sum(shares.get(h, 0) for h in homes), decimals)

    rows = []
    for m in methods:
        alone, family = to_shares(alone_homes[m]), to_shares(family_homes[m])
        rows.append({
            "method": m,
            "retired_alone_with_income_1k_plus_pct": round(100 * retired_alone_earning[m] / retired_alone[m], 2),
            "alone_seniors_in_1_3_room_pct": pct_in(alone, SMALL_FLATS),
            "alone_vs_national_1_person_tvd": round(compare(alone, ref_alone)[2], 3),
            "family_seniors_in_4_room_plus_pct": pct_in(family, BIGGER_HOMES),
            "family_vs_national_3plus_tvd": round(compare(family, ref_family)[2], 3),
        })
    rows.append({
        "method": "national reference",
        "retired_alone_with_income_1k_plus_pct": "",
        "alone_seniors_in_1_3_room_pct": pct_in(ref_alone, SMALL_FLATS),
        "alone_vs_national_1_person_tvd": 0.0,
        "family_seniors_in_4_room_plus_pct": pct_in(ref_family, BIGGER_HOMES),
        "family_vs_national_3plus_tvd": 0.0,
    })
    return rows


# ---------------------------------------------------------------------------
# Check 7: same seed -> same personas
# ---------------------------------------------------------------------------

def check_reproducibility(towns):
    def draw(town, n, seed):
        return [p.model_dump() for p in new_sampler.sample_synthetic_cohort(town, n, seed=seed)]

    rows = []
    for town in towns:
        first = draw(town, 30, 42)
        rows.append({"area": town,
                     "same_seed_identical": first == draw(town, 30, 42),
                     "different_seed_differs": first != draw(town, 30, 43),
                     "first_30_kept_when_n_100": draw(town, 100, 42)[:30] == first})
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(title, rows, group_by):
    """One line per table: median and worst result across towns."""
    print(f"\n## {title}")
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_by)].append(r)
    print(" | ".join([*group_by, "towns", "median |gap| pts", "worst gap pts (town, category)",
                      "median TVD", "median SRMSE"]))
    for key, rs in groups.items():
        worst = max(rs, key=lambda r: abs(r["max_gap_pts"]))
        print(" | ".join([*map(str, key), str(len(rs)),
                          f"{np.median([abs(r['max_gap_pts']) for r in rs]):.2f}",
                          f"{worst['max_gap_pts']:+.2f} ({worst['area']}, {worst['category']})",
                          f"{np.median([r['tvd'] for r in rs]):.4f}",
                          f"{np.median([r['srmse'] for r in rs]):.3f}"]))


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=40000, help="personas per town for the mix checks")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--panels", type=int, default=500, help="30-persona panels per town for the noise check")
    args = parser.parse_args()
    towns = new_sampler.list_supported_areas()
    print(f"E1 sampler fidelity: {len(towns)} towns, {args.n:,} personas per town, seed {args.seed}")

    # 1 and 1b
    rows = check_census_fit()
    save_csv("1_census_fit", rows)
    print_summary("1. Census fit (fit.py) vs published town totals", rows, ("table",))
    rows = check_joint_vs_national()
    save_csv("1b_joint_vs_national", rows)
    print_summary("1b. Income within each home type, all towns added up, vs national table", rows, ("table",))

    # 2: new sampler, all towns, detailed groups
    rows = check_persona_mix("census sampler", new_sampler.sample_synthetic_cohort, towns, args.n, args.seed,
                             ["dwelling", "income_band", "age_sex", "sex"])
    save_csv("2_persona_mix", rows)
    print_summary("2. Persona mix vs census (all towns)", rows, ("table",))

    # 4: chance alone in a 30-persona panel
    rows = check_panel_noise(CHECK_TOWNS, args.panels)
    save_csv("4_panel_noise", rows)
    print("\n## 4. One 30-persona panel vs census (sampling noise, new sampler)")
    for table in ("dwelling", "income_coarse", "age_coarse"):
        rs = [r for r in rows if r["table"] == table]
        print(f"{table}: mean TVD {np.mean([r['mean_tvd'] for r in rs]):.3f}, "
              f"90th percentile {np.mean([r['p90_tvd'] for r in rs]):.3f} (average over {len(rs)} towns)")

    # 5: seniors, all towns pooled. Each town gets its own seed: with one shared seed the towns
    # would reuse the same random numbers, and the pooled personas would not be independent.
    pooled = []
    for i, town in enumerate(towns):
        pooled += new_sampler.sample_synthetic_cohort(town, args.n // 4, seed=args.seed + i)
    rows = check_seniors(pooled)
    save_csv("5_senior_details", rows)
    print(f"\n## 5. Senior details vs national 2025 tables ({len(pooled):,} pooled personas)")
    for check in ("labour_force_rate", "living_arrangement"):
        rs = [r for r in rows if r["check"] == check]
        worst = max(rs, key=lambda r: abs(r["gap_pts"]))
        print(f"{check}: {len(rs)} cells, median |gap| {np.median([abs(r['gap_pts']) for r in rs]):.2f} pts, "
              f"worst {worst['gap_pts']:+.2f} pts ({worst['group']}, {worst['personas']} personas)")

    # 6: step 14 before vs after
    rows = check_household_plausibility(towns, args.n // 4, args.seed)
    save_csv("6_household_plausibility", rows)
    print("\n## 6. Household plausibility: step 14 vs independent home draw (all towns, population-weighted)")
    for r in rows:
        print(r)

    # 7: reproducibility
    rows = check_reproducibility(CHECK_TOWNS)
    save_csv("7_reproducibility", rows)
    print("\n## 7. Reproducibility")
    for check in ("same_seed_identical", "different_seed_differs", "first_30_kept_when_n_100"):
        print(f"{check}: {sum(r[check] for r in rows)}/{len(rows)} towns")
    print(f"\nCSVs written to {OUT}")


if __name__ == "__main__":
    main()
