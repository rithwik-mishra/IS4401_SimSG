"""
SimSG - clean the source tables into tidy CSV tables.

Run from backend/:
    python pipeline/clean_data.py

Folder layout (under data/sheets/):
    raw/         original downloads. Two of them are read directly, since SingStat / MOM already
                 publish them as machine-readable CSV (no hand-tidied copy needed):
                   2025_Resident_LabourForce_AgeGroup_Gender.csv    resident labour force 2025, by age x sex
                                                                    (x marital status x ethnic group; Totals used)
                   Resident65Plus_LivingArrangement_AgeGroup.csv    residents 65+ by living arrangement x age group
                                                                    (annual series; 2025 column used)
    cleaned/     the seven Cleaned_*.xlsx workbooks  -> read, never modified
    processed/   tidy CSV tables                     -> written by this script
    checks/      cleaning_report.txt, published_area_totals.csv

Every output table is in long format: one row per combination, one count column.
The two senior tables (labour force rate, living arrangement) are national, not by area,
so pipeline/fit.py does not use them; the sampler reads them directly.
"""

import re
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "sheets"
RAW = DATA / "cleaned"
DOWNLOADS = DATA / "raw"  # original downloads; the two CSVs below are read straight from here
CLEAN = DATA / "processed"
CHECKS = DATA / "checks"

RAW_CSV_FILES = {
    "labour_force": "2025_Resident_LabourForce_AgeGroup_Gender.csv",
    "living": "Resident65Plus_LivingArrangement_AgeGroup.csv",
}
LIVING_YEAR = "2025"
LABOUR_FORCE_TOP_AGE = 70  # the labour force table's oldest group is "70 & Over"

FILES = {
    "ethnic": "Cleaned_2025_Resident_Area_EthnicGroup_Gender.xlsx",
    "age": "Cleaned_2025_Resident_Area_AgeGroup_Gender.xlsx",
    "area_income": "Cleaned_2025_Household_Area_MonthlyIncome.xlsx",
    "income_dwelling": "Cleaned_2025_Household_MonthlyIncome_DwellingType.xlsx",
    "area_dwelling": "Cleaned_2025_Household_Area_DwellingType.xlsx",
    "spend": "Cleaned_2023_Resident_AverageExpenditure_DwellingType.xlsx",
    "size": "Cleaned_2025_Household_Area_HouseholdSize_Tenancy.xlsx",
}

DROP_AREAS = {"Total", "Others"}  # national total row; household-only grouping of 26 small areas

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


# ---------------------------------------------------------------------------
# Label standardisation (the single mapping used by every table)
# ---------------------------------------------------------------------------

def strip_footnote(text):
    """'HDB 1- & 2-Room Flats3/' -> 'HDB 1- & 2-Room Flats'; also trims spaces."""
    return re.sub(r"\s*\d+/\s*$", "", str(text)).strip()


DWELLING_KEYS = [
    ("1-2 room", "HDB 1- & 2-Room Flats"),
    ("3 room", "HDB 3-Room Flats"),
    ("4 room", "HDB 4-Room Flats"),
    ("5 room/exec", "HDB 5-Room & Executive Flats"),
    ("condo", "Condominiums & Other Apartments"),
    ("landed", "Landed Properties"),
]
DWELLING_ORDER = [k for k, _ in DWELLING_KEYS]


def dwelling_key(label):
    """Map any spelling of a dwelling type to one short key, or None for totals."""
    s = strip_footnote(label).lower().replace("hdb ", "").replace(" ", "")
    if s in {"total", "dwellings"}:
        return None
    if s.startswith("1-&2-room"):
        return "1-2 room"
    if s.startswith("3-room"):
        return "3 room"
    if s.startswith("4-room"):
        return "4 room"
    if s.startswith("5-room"):
        return "5 room/exec"
    if s.startswith("condominium"):
        return "condo"
    if s.startswith("landed"):
        return "landed"
    raise ValueError(f"Unknown dwelling label: {label!r}")


def income_band(label):
    """'$25,000 -$29,999' / '25,000 - 29,999' -> ('25000-29999', 25000, 29999)."""
    s = re.sub(r"[\s$,]", "", strip_footnote(label))
    if s.lower().startswith("below"):
        hi = int(re.sub(r"\D", "", s))
        return f"0-{hi - 1}", 0, hi - 1
    if s.lower().endswith("&over"):
        lo = int(re.sub(r"\D", "", s))
        return f"{lo}+", lo, None
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise ValueError(f"Unknown income band: {label!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    return f"{lo}-{hi}", lo, hi


def age_group(label):
    """'0 - 4' -> ('0-4', 0); '90 & Over' -> ('90+', 90)."""
    s = re.sub(r"\s", "", str(label))
    if s.lower().endswith("&over"):
        lo = int(re.sub(r"\D", "", s))
        return f"{lo}+", lo
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise ValueError(f"Unknown age group: {label!r}")
    return f"{m.group(1)}-{m.group(2)}", int(m.group(1))


def to_number(series):
    """'-' means nil in these tables, so it becomes 0. Anything else unreadable stays missing."""
    s = series.replace({"-": 0, "–": 0})
    return pd.to_numeric(s, errors="coerce")


def read(key, sheet=0):
    return pd.read_excel(RAW / FILES[key], sheet_name=sheet)


def tidy_areas(df):
    df = df.rename(columns={df.columns[0]: "area"})
    df["area"] = df["area"].astype(str).str.strip()
    return df[~df["area"].isin(DROP_AREAS)].copy()


# ---------------------------------------------------------------------------
# 1. Residents: area x sex x ethnic group
# ---------------------------------------------------------------------------

def clean_ethnic():
    parts = []
    for group in ["Chinese", "Malay", "Indian", "Others"]:
        df = tidy_areas(read("ethnic", group)).drop(columns=["Total"])
        df = df.melt(id_vars="area", var_name="sex", value_name="residents")
        df["ethnic_group"] = group
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["residents"] = to_number(out["residents"])
    return out[["area", "sex", "ethnic_group", "residents"]]


# ---------------------------------------------------------------------------
# 2. Residents: area x age group x sex
# ---------------------------------------------------------------------------

def clean_age():
    parts = []
    for tab, sex in [("Males", "Male"), ("Females", "Female")]:
        df = tidy_areas(read("age", tab)).drop(columns=["Total"])
        df = df.melt(id_vars="area", var_name="age_raw", value_name="residents")
        df["sex"] = sex
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out[["age_group", "age_lo"]] = out["age_raw"].apply(lambda x: pd.Series(age_group(x)))
    out["residents"] = to_number(out["residents"])
    return out[["area", "age_group", "age_lo", "sex", "residents"]]


# ---------------------------------------------------------------------------
# 3. Households: area x monthly household market income
# ---------------------------------------------------------------------------

def clean_area_income():
    df = tidy_areas(read("area_income"))
    published = df.set_index("area")["Total"]
    df = df.drop(columns=["Total"]).melt(id_vars="area", var_name="band_raw", value_name="households")
    df[["income_band", "income_lo", "income_hi"]] = df["band_raw"].apply(lambda x: pd.Series(income_band(x)))
    df["households"] = to_number(df["households"])
    return df[["area", "income_band", "income_lo", "income_hi", "households"]], published


# ---------------------------------------------------------------------------
# 4. Households: income band x dwelling (national seed)
# ---------------------------------------------------------------------------

def clean_income_dwelling():
    df = read("income_dwelling")
    df = df.rename(columns={df.columns[0]: "band_raw"})
    df = df[df["band_raw"].astype(str).str.strip() != "Total"].drop(columns=["Total"])
    df = df.melt(id_vars="band_raw", var_name="dwelling_raw", value_name="households")
    df[["income_band", "income_lo", "income_hi"]] = df["band_raw"].apply(lambda x: pd.Series(income_band(x)))
    df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
    df["households"] = to_number(df["households"])
    return df[["income_band", "income_lo", "income_hi", "dwelling", "households"]]


# ---------------------------------------------------------------------------
# 5. Households: area x dwelling
# ---------------------------------------------------------------------------

def clean_area_dwelling():
    df = tidy_areas(read("area_dwelling"))
    published = df.set_index("area")["Total"]
    df = df.drop(columns=["Total"]).melt(id_vars="area", var_name="dwelling_raw", value_name="households")
    df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
    df["households"] = to_number(df["households"])
    return df[["area", "dwelling", "households"]], published


# ---------------------------------------------------------------------------
# 6. Spending per household member x dwelling (HES 2023, S$ per month)
# ---------------------------------------------------------------------------

def clean_spend():
    df = read("spend")
    df = df.rename(columns={df.columns[0]: "category"})
    df["category"] = df["category"].map(strip_footnote)
    df = df.drop(columns=["Total"]).melt(id_vars="category", var_name="dwelling_raw", value_name="spend_per_member")
    df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
    df["spend_per_member"] = to_number(df["spend_per_member"])
    return df[["dwelling", "category", "spend_per_member"]]


# ---------------------------------------------------------------------------
# 7. Households: dwelling x household size (national), with tenure split
# ---------------------------------------------------------------------------

SIZE_TABS = {"1_person": 1, "2_person": 2, "3_person": 3, "4_person": 4,
             "5_person": 5, "6_people_or_more": 6}  # 6 means "6 or more"


def clean_size():
    parts = []
    for tab, size in SIZE_TABS.items():
        df = read("size", tab)
        df = df.rename(columns={df.columns[0]: "dwelling_raw",
                                "Total": "households", "Owner-Occupied": "owner_occupied", "Rented": "rented"})
        df["dwelling"] = df["dwelling_raw"].map(dwelling_key)
        df = df[df["dwelling"].notna()].copy()  # drops 'Total' and 'HDB Dwellings' subtotal rows
        df["household_size"] = size
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    for c in ["households", "owner_occupied", "rented"]:
        out[c] = to_number(out[c])
    return out[["dwelling", "household_size", "households", "owner_occupied", "rented"]]


# ---------------------------------------------------------------------------
# 8. Residents: labour force rate by age group x sex (national, 2025)
# ---------------------------------------------------------------------------

def national_population_age_sex():
    """National residents by age group x sex: the 'Total' row of the age workbook (dropped from the area tables)."""
    parts = []
    for tab, sex in [("Males", "Male"), ("Females", "Female")]:
        df = read("age", tab)
        df = df.rename(columns={df.columns[0]: "area"})
        total = df[df["area"].astype(str).str.strip() == "Total"].drop(columns=["area", "Total"])
        long = total.melt(var_name="age_raw", value_name="population")
        long["sex"] = sex
        parts.append(long)
    out = pd.concat(parts, ignore_index=True)
    out[["age_group", "age_lo"]] = out["age_raw"].apply(lambda x: pd.Series(age_group(x)))
    out["population"] = to_number(out["population"])
    return out[["age_group", "age_lo", "sex", "population"]]


def clean_labour_force():
    """
    Resident labour force (working or looking for work) by age group x sex, divided by the national
    population of the same age and sex. The 'Total' ethnic group and 'Total' marital status columns
    are used. Population ages 70+ are summed to match the labour force table's '70 & Over' group.
    The two tables come from different surveys, so a few prime-age rates exceed 100% (see checks);
    the sampler only uses ages 60 and over.
    """
    raw = pd.read_csv(DOWNLOADS / RAW_CSV_FILES["labour_force"], header=None, dtype=str)
    header = [str(raw.iloc[r, c]).strip() for r, c in [(0, 1), (1, 1), (2, 1), (2, 2), (2, 3)]]
    if header != ["Total", "Total", "Total", "Males", "Females"]:
        raise ValueError(f"Unexpected labour force header (columns 1-3 should be Total/Total/Total,Males,Females): {header}")

    body = raw.iloc[3:, :4].dropna(subset=[0])
    body.columns = ["age_raw", "total", "Male", "Female"]
    body["age_raw"] = body["age_raw"].str.strip()
    body = body[body["age_raw"] != "Total"]
    lf = body.melt(id_vars="age_raw", value_vars=["Male", "Female"], var_name="sex", value_name="labour_force")
    lf[["age_group", "age_lo"]] = lf["age_raw"].apply(lambda x: pd.Series(age_group(x)))
    lf["labour_force"] = to_number(lf["labour_force"].str.replace(",", "", regex=False))

    pop = national_population_age_sex()
    pop["age_group"] = pop.apply(
        lambda r: f"{LABOUR_FORCE_TOP_AGE}+" if r["age_lo"] >= LABOUR_FORCE_TOP_AGE else r["age_group"], axis=1)
    pop["age_lo"] = pop["age_lo"].clip(upper=LABOUR_FORCE_TOP_AGE)
    pop = pop.groupby(["age_group", "age_lo", "sex"], as_index=False)["population"].sum()

    out = lf.merge(pop, on=["age_group", "age_lo", "sex"], how="left")
    out["rate"] = out["labour_force"] / out["population"]
    out = out.sort_values(["age_lo", "sex"]).reset_index(drop=True)
    return out[["age_group", "age_lo", "sex", "labour_force", "population", "rate"]]


# ---------------------------------------------------------------------------
# 9. Residents 65+: living arrangement by age group (national, 2025)
# ---------------------------------------------------------------------------

LIVING_ARRANGEMENTS = [
    # (key, label prefix in the source table, wording used in the persona prompt)
    ("spouse_and_child", "living with spouse & at least one child", "lives with spouse and at least one child"),
    ("spouse_only", "living with spouse, no children", "lives with spouse, no children in the household"),
    ("child_no_spouse", "living with children but without spouse", "lives with child(ren), without a spouse"),
    ("alone", "living in one-person household", "lives alone"),
    ("domestic_worker_only", "living with only domestic worker", "lives with only a domestic worker"),
    ("others", "others", "lives with other relatives or others"),
]


def living_key(label):
    s = str(label).strip().lower()
    for key, prefix, _ in LIVING_ARRANGEMENTS:
        if s.startswith(prefix):
            return key
    raise ValueError(f"Unknown living arrangement: {label!r}")


def living_age_group(label):
    """'65 - 69 Years' -> ('65-69', 65); '85 Years & Over' -> ('85+', 85)."""
    return age_group(re.sub(r"(?i)\s*years?\s*", " ", str(label)).strip())


def clean_living_arrangement():
    """
    Residents aged 65+ in resident households by living arrangement x age group, for LIVING_YEAR.
    Counts people (not households), so it is not tied to the age of the household head.
    Group rows (no indent) are the age totals; indented rows are the arrangements within them.
    """
    df = pd.read_csv(DOWNLOADS / RAW_CSV_FILES["living"], dtype=str)
    if LIVING_YEAR not in df.columns:
        raise ValueError(f"Column {LIVING_YEAR} not found in {RAW_CSV_FILES['living']}")
    rows, published, current = [], {}, None
    for label, value in zip(df.iloc[:, 0], df[LIVING_YEAR]):
        if pd.isna(label):
            continue
        is_group = not str(label).startswith(" ")
        count = float(str(value).replace(",", ""))
        if is_group:
            current = None if str(label).strip() == "Total" else living_age_group(label)
            if current:
                published[current[0]] = count
            continue
        if current is None:
            continue  # arrangements of the all-ages 'Total' group
        rows.append({"age_group": current[0], "age_lo": current[1], "arrangement": living_key(label),
                     "residents": count})
    out = pd.DataFrame(rows)
    texts = {key: text for key, _, text in LIVING_ARRANGEMENTS}
    out["arrangement_text"] = out["arrangement"].map(texts)
    out["share"] = out["residents"] / out.groupby("age_group")["residents"].transform("sum")
    out = out.sort_values(["age_lo", "arrangement"]).reset_index(drop=True)
    return out[["age_group", "age_lo", "arrangement", "arrangement_text", "residents", "share"]], published


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(name, ok, detail=""):
    log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def run_checks(t, published_income, published_dwelling):
    results = []
    areas = {k: set(t[k]["area"]) for k in ["ethnic", "age", "area_income", "area_dwelling"]}
    ref = areas["area_dwelling"]
    results.append(check("Same areas in all four area tables",
                         all(a == ref for a in areas.values()), f"{len(ref)} areas"))

    for k in ["income_dwelling", "area_dwelling", "spend", "size"]:
        got = set(t[k]["dwelling"])
        results.append(check(f"Dwelling labels match in {k}", got == set(DWELLING_ORDER), str(sorted(got))))

    b1, b2 = set(t["area_income"]["income_band"]), set(t["income_dwelling"]["income_band"])
    results.append(check("Income bands match between area x income and income x dwelling",
                         b1 == b2, f"{len(b1)} bands"))

    for k, col in [("ethnic", "residents"), ("age", "residents"), ("area_income", "households"),
                   ("income_dwelling", "households"), ("area_dwelling", "households"),
                   ("spend", "spend_per_member"), ("size", "households")]:
        n = int(t[k][col].isna().sum())
        neg = int((t[k][col] < 0).sum())
        results.append(check(f"No missing or negative values in {k}", n == 0 and neg == 0,
                             f"missing={n}, negative={neg}"))

    e = t["ethnic"].groupby(["area", "sex"])["residents"].sum()
    a = t["age"].groupby(["area", "sex"])["residents"].sum()
    gap = (e - a).abs().max()
    results.append(check("Area x sex totals agree between ethnic and age tables (rounding <= 50)",
                         gap <= 50, f"max gap {gap:.0f}"))

    inc = t["area_income"].groupby("area")["households"].sum()
    dwe = t["area_dwelling"].groupby("area")["households"].sum()
    g1 = (inc - published_income).abs().max()
    g2 = (dwe - published_dwelling).abs().max()
    results.append(check("Area income rows close to published area totals (rounding <= 500)",
                         g1 <= 500, f"max gap {g1:.0f}"))
    # The published area total also counts households in dwelling types that have no column
    # (about 3,500 nationally), so the six dwelling columns can fall short of it.
    short = published_dwelling - dwe
    log(f"[INFO] Households in unlisted dwelling types per area: up to {short.max():.0f} "
        f"(largest: {short.idxmax()}); total across 29 areas {short.sum():.0f}")
    results.append(check("Dwelling rows never exceed the published area total beyond rounding (<= 200)",
                         short.min() >= -200, f"min {short.min():.0f}"))
    results.append(check("Published area totals identical in income and dwelling tables",
                         (published_income - published_dwelling).abs().max() == 0))

    nat_size = t["size"].groupby("dwelling")["households"].sum()
    nat_dw = t["area_dwelling"].groupby("dwelling")["households"].sum()
    log(f"[INFO] Size table national households: {int(nat_size.sum()):,}; "
        f"29-area dwelling table: {int(nat_dw.sum()):,} (difference is the dropped 'Others' row + rounding)")

    fnb = t["spend"].query("category == 'Food and Beverage Serving Services'")
    results.append(check("Spending table has Food and Beverage Serving Services for all 6 dwellings",
                         len(fnb) == 6))
    return all(results)


def run_senior_checks(lf, living, living_published):
    """Checks for the two national senior tables (labour force rate, living arrangement)."""
    results = []
    results.append(check("Labour force table has 12 age groups x 2 sexes, no missing values",
                         len(lf) == 24 and not lf[["labour_force", "population", "rate"]].isna().any().any(),
                         f"{len(lf)} rows"))
    nat = national_population_age_sex()
    aged_15_plus = nat.query("age_lo >= 15")["population"].sum()
    results.append(check("Population used for the rates = all residents aged 15+ (age workbook 'Total' row)",
                         abs(aged_15_plus - lf["population"].sum()) < 1, f"{int(aged_15_plus):,} residents aged 15+"))
    senior = lf[lf["age_lo"] >= 60]
    results.append(check("Labour force rates for ages 60+ (the ones the sampler uses) are between 0% and 100%",
                         bool(((senior["rate"] >= 0) & (senior["rate"] <= 1)).all()),
                         ", ".join(f"{r.age_group} {r.sex[0]} {r.rate:.1%}" for r in senior.itertuples())))
    over = lf[lf["rate"] > 1]
    log(f"[INFO] Labour force rates above 100% (labour force and population tables come from different "
        f"surveys; not used by the sampler): "
        + (", ".join(f"{r.age_group} {r.sex} {r.rate:.1%}" for r in over.itertuples()) or "none"))

    sums = living.groupby("age_group")["residents"].sum()
    gaps = {g: abs(sums[g] - living_published[g]) for g in living_published}
    results.append(check("Living arrangements add up to each published age-group total (rounding <= 300)",
                         set(sums.index) == set(living_published) and max(gaps.values()) <= 300,
                         f"max gap {max(gaps.values()):.0f}"))
    results.append(check("Living arrangement table has 5 age groups x 6 arrangements, shares sum to 1",
                         len(living) == 30 and bool((living.groupby("age_group")["share"].sum().sub(1).abs() < 1e-9).all())))
    return all(results)


# ---------------------------------------------------------------------------

def main():
    missing = [str(RAW / f) for f in FILES.values() if not (RAW / f).exists()]
    missing += [str(DOWNLOADS / f) for f in RAW_CSV_FILES.values() if not (DOWNLOADS / f).exists()]
    if missing:
        raise SystemExit("Not found:\n  " + "\n  ".join(missing))
    CLEAN.mkdir(parents=True, exist_ok=True)
    CHECKS.mkdir(parents=True, exist_ok=True)

    tables = {}
    tables["ethnic"] = clean_ethnic()
    tables["age"] = clean_age()
    tables["area_income"], published_income = clean_area_income()
    tables["income_dwelling"] = clean_income_dwelling()
    tables["area_dwelling"], published_dwelling = clean_area_dwelling()
    tables["spend"] = clean_spend()
    tables["size"] = clean_size()
    tables["labour_force"] = clean_labour_force()
    tables["living"], living_published = clean_living_arrangement()

    names = {
        "ethnic": "resident_area_sex_ethnic.csv",
        "age": "resident_area_age_sex.csv",
        "area_income": "household_area_income.csv",
        "income_dwelling": "household_income_dwelling.csv",
        "area_dwelling": "household_area_dwelling.csv",
        "spend": "spend_per_member_dwelling.csv",
        "size": "household_dwelling_size.csv",
        "labour_force": "labour_force_rate_age_sex.csv",
        "living": "senior_living_arrangement_age.csv",
    }
    for k, df in tables.items():
        df.to_csv(CLEAN / names[k], index=False)
        log(f"Wrote data/processed/{names[k]}  ({len(df)} rows)")

    # Keep the published totals as a reference file (used for checks, never for sampling).
    res = read("ethnic", "Total").rename(columns=lambda c: c.strip())
    res = res.rename(columns={res.columns[0]: "area"})
    res["area"] = res["area"].astype(str).str.strip()
    res = res.set_index("area")
    ref = pd.DataFrame({
        "households_published": published_dwelling,
        "households_in_6_dwelling_types": tables["area_dwelling"].groupby("area")["households"].sum(),
        "residents_published": res.loc[published_dwelling.index, "Total"],
        "residents_male": res.loc[published_dwelling.index, "Male"],
        "residents_female": res.loc[published_dwelling.index, "Female"],
    })
    ref.index.name = "area"
    ref.to_csv(CHECKS / "published_area_totals.csv")
    log("Wrote data/checks/published_area_totals.csv  (reference only)")

    log("")
    ok = run_checks(tables, published_income, published_dwelling)
    ok = run_senior_checks(tables["labour_force"], tables["living"], living_published) and ok
    log("")
    log("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED - fix before building the fit")
    (CHECKS / "cleaning_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()