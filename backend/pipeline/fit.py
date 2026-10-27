"""
SimSG - build a dwelling x income table for each planning area (iterative proportional fitting).

Run after clean_data.py, from anywhere:
    python fit.py

For each of the 29 areas:
    seed           national income x dwelling households     (household_income_dwelling.csv)
    row targets    the area's households by dwelling         (household_area_dwelling.csv)
    column targets the area's households by income band      (household_area_income.csv),
                   scaled to the row total (it also counts homes outside the 6 dwelling types)
The fit keeps the national dwelling-income pattern and matches both of the area's totals.

Each fitted cell is then multiplied by the average household size of its dwelling type
(household_dwelling_size.csv), turning household counts into resident weights for sampling.

Writes:
    data/processed/fitted_area_dwelling_income.csv   one row per area x dwelling x income band
    data/checks/fit_report.csv                       convergence and error per area
"""

from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
DATA = BACKEND / "data" / "sheets"
PROC = DATA / "processed"
CHECKS = DATA / "checks"

TOP_SIZE = 6.5      # value assumed for the "6 or more" household-size group
MAX_ITERS = 500
TOL = 0.01          # households; stop when every row and column is within this of its target

DWELLINGS = ["1-2 room", "3 room", "4 room", "5 room/exec", "condo", "landed"]


def ipf(seed, row_t, col_t):
    """Scale rows then columns until both match their targets. Returns (table, iterations, converged)."""
    t = seed.astype(float).copy()
    for i in range(1, MAX_ITERS + 1):
        rs = t.sum(axis=1)
        t *= np.divide(row_t, rs, out=np.zeros_like(rs), where=rs > 0)[:, None]
        cs = t.sum(axis=0)
        t *= np.divide(col_t, cs, out=np.zeros_like(cs), where=cs > 0)[None, :]
        if np.abs(t.sum(axis=1) - row_t).max() < TOL and np.abs(t.sum(axis=0) - col_t).max() < TOL:
            return t, i, True
    return t, MAX_ITERS, False


def main():
    seed_long = pd.read_csv(PROC / "household_income_dwelling.csv")
    area_dw = pd.read_csv(PROC / "household_area_dwelling.csv")
    area_inc = pd.read_csv(PROC / "household_area_income.csv")
    size = pd.read_csv(PROC / "household_dwelling_size.csv")

    bands = (seed_long.drop_duplicates("income_band")
             .sort_values("income_lo")[["income_band", "income_lo", "income_hi"]])
    band_order = bands["income_band"].tolist()

    seed = (seed_long.pivot(index="dwelling", columns="income_band", values="households")
            .reindex(index=DWELLINGS, columns=band_order).fillna(0).to_numpy())

    # Average household size per dwelling type (national)
    size["size_value"] = size["household_size"].replace({6: TOP_SIZE})
    avg_size = (size.assign(people=size["size_value"] * size["households"])
                .groupby("dwelling")[["people", "households"]].sum())
    avg_size = (avg_size["people"] / avg_size["households"]).reindex(DWELLINGS)

    rows, report = [], []
    for area in sorted(area_dw["area"].unique()):
        row_t = (area_dw[area_dw["area"] == area].set_index("dwelling")["households"]
                 .reindex(DWELLINGS).fillna(0).to_numpy(float))
        col_raw = (area_inc[area_inc["area"] == area].set_index("income_band")["households"]
                   .reindex(band_order).fillna(0).to_numpy(float))
        col_t = col_raw * row_t.sum() / col_raw.sum()     # both margins must share one total

        fitted, iters, ok = ipf(seed, row_t, col_t)

        # Cells the margins demand but the seed cannot supply (seed zero) would show up here
        row_err = np.abs(fitted.sum(axis=1) - row_t).max()
        col_err = np.abs(fitted.sum(axis=0) - col_t).max()
        report.append({
            "area": area, "converged": ok, "iterations": iters,
            "max_row_error": round(row_err, 4), "max_col_error": round(col_err, 4),
            "households_in_6_types": int(row_t.sum()),
            "income_total_before_scaling": int(col_raw.sum()),
            "dwelling_types_present": int((row_t > 0).sum()),
            "largest_cell_share": round(fitted.max() / fitted.sum(), 4),
        })

        for i, dw in enumerate(DWELLINGS):
            for j, band in enumerate(band_order):
                rows.append({"area": area, "dwelling": dw, "income_band": band,
                             "households_fitted": fitted[i, j]})

    out = pd.DataFrame(rows).merge(bands, on="income_band")
    out["avg_household_size"] = out["dwelling"].map(avg_size)
    out["resident_weight"] = out["households_fitted"] * out["avg_household_size"]
    out["prob_in_area"] = out["resident_weight"] / out.groupby("area")["resident_weight"].transform("sum")
    out = out[out["households_fitted"] > 0]   # drop impossible cells to keep the file small
    out = out[["area", "dwelling", "income_band", "income_lo", "income_hi",
               "households_fitted", "avg_household_size", "resident_weight", "prob_in_area"]]
    out.to_csv(PROC / "fitted_area_dwelling_income.csv", index=False)

    rep = pd.DataFrame(report)
    pub = pd.read_csv(CHECKS / "published_area_totals.csv").set_index("area")
    est = out.groupby("area")["resident_weight"].sum()
    rep["estimated_residents"] = rep["area"].map(est).round()
    rep["published_residents"] = rep["area"].map(pub["residents_published"])
    rep["resident_ratio"] = (rep["estimated_residents"] / rep["published_residents"]).round(3)
    rep.to_csv(CHECKS / "fit_report.csv", index=False)

    print("Average household size used:", avg_size.round(2).to_dict())
    print(f"Wrote data/processed/fitted_area_dwelling_income.csv  ({len(out)} cells, {out['area'].nunique()} areas)")
    print("Wrote data/checks/fit_report.csv")
    print()
    print(f"Converged: {int(rep['converged'].sum())}/{len(rep)} areas, "
          f"max iterations {rep['iterations'].max()}, "
          f"max row error {rep['max_row_error'].max()}, max column error {rep['max_col_error'].max()}")
    print("Probabilities sum to 1 in every area:",
          bool(np.allclose(out.groupby("area")["prob_in_area"].sum(), 1)))
    print(f"Estimated / published residents: min {rep['resident_ratio'].min()} "
          f"({rep.loc[rep['resident_ratio'].idxmin(), 'area']}), "
          f"max {rep['resident_ratio'].max()} ({rep.loc[rep['resident_ratio'].idxmax(), 'area']})")
    if not rep["converged"].all():
        print("NOT CONVERGED:", rep.loc[~rep["converged"], "area"].tolist())
    print("FIT OK" if rep["converged"].all() else "FIT HAS PROBLEMS - see data/checks/fit_report.csv")


if __name__ == "__main__":
    main()