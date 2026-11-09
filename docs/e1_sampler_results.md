# E1 results: persona sampler fidelity

**Question:** do SimSG's synthetic personas look like the residents of each town, as described by the
census?

**Ground truth:** SingStat census tables in `backend/data/sheets/processed/` (residents by age and sex;
households by home type and by monthly household market income, per town; 2025 national labour force and living
arrangement tables). Personas are adults (20+), so home type and income are weighted by residents.

**How to reproduce** (from `backend/`, about 1.5 minutes):
```
python scripts/evaluate_sampler.py            # 40,000 personas per town, seed 2026
```
Full tables: `backend/data/evaluation/e1/*.csv`.

**Metrics:**
- **Largest gap:** the biggest difference in percentage points between the persona share and the census share, over all categories of a table.
- **TVD (total variation distance):** half the sum of absolute differences. It's the share of personas that would have to change category to match the census exactly (0 = perfect, 1 = completely different).
- **SRMSE:** root mean squared error ÷ mean share, the standard fit measure in population synthesis (0 = perfect).

---

## 1. Census fit (fit.py)

Each town's fitted home type × income table reproduces its published totals for home type and for
income **exactly** (gap 0.00 points, all 29 towns). That's expected, since the fitting method forces it.

Added up over all towns, the income mix **within** each home type stays close to the national table:

| Home type | TVD | Largest gap |
| --- | --- | --- |
| 1-2 room | 0.015 | −0.9 pts (below S$1,000) |
| 3 room | 0.017 | +0.7 pts (below S$1,000) |
| 4 room | 0.007 | −0.3 pts (S$35,000+) |
| 5 room/exec | 0.019 | −1.2 pts (S$35,000+) |
| condo | 0.010 | +0.6 pts (S$35,000+) |
| landed | 0.030 | +2.3 pts (S$35,000+) |

This is a consistency check, not independent proof: the fit starts from the national table, and no
town-level home type × income table is published to test against.

## 2. Persona mix vs census (29 towns, 40,000 personas each)

| Table | Median largest gap | Worst town | Median TVD | Median SRMSE |
| --- | --- | --- | --- | --- |
| Home type | 0.35 pts | −0.68 pts (Outram, 3 room) | 0.005 | 0.012 |
| Income band (19 bands) | 0.20 pts | +2.39 pts (Tanglin) | 0.008 | 0.019 |
| Age group × sex (28 cells) | 0.23 pts | −0.35 pts (Tengah) | 0.011 | 0.028 |
| Sex | 0.31 pts | +0.35 pts (Bukit Timah) | 0.003 | 0.006 |

Two known exceptions, both caused by the senior household rules (step 14):
- **Tanglin** has no "below S$1,000" households in the census, so retired seniors living alone get the lowest band that exists there (S$3,000–3,999). That band runs 2.4 points high.
- **Bukit Timah:** national rates predict more retired seniors living alone (2.7%) than the town's census "below S$1,000" share (1.1%) can hold, so that band runs 1.7 points high.

## 4. How far one 30-persona panel is from the census (sampling noise)

A real run uses 30 personas, so random variation matters more than sampler bias. Over 500 panels
per town (6 towns):

| Table | Mean TVD of a 30-persona panel | 90th percentile |
| --- | --- | --- |
| Home type | 0.137 | 0.205 |
| Income (6 groups) | 0.150 | 0.221 |
| Age (5 groups) | 0.141 | 0.213 |

So about 14% of a typical 30-persona panel is "off-mix" by chance alone. For home type (the same
six groups in both checks) that is 14–34 times the sampler's own error in these towns (TVD
0.004–0.010 in check 2). The panel size, not the sampler, limits how representative one run is. This
links to E4g (stability across seeds) and is a reason to report results with confidence intervals.

## 5. Senior details vs national 2025 tables (290,000 pooled personas)

| Check | Cells | Median gap | Worst gap |
| --- | --- | --- | --- |
| Labour force rate (60–64 to 70+, by sex) | 6 | 0.43 pts | +0.87 pts (60–64 female) |
| Living arrangement (65–69 to 85+, 6 types) | 30 | 0.21 pts | +0.94 pts (85+, spouse only) |

## 6. Household plausibility: step 14 vs drawing the home independently of the person

Same people, two ways of drawing home type and income (all towns, weighted by town population):

| Measure | Step 14 (person first) | Independent draw (before step 14) | National reference |
| --- | --- | --- | --- |
| Retired seniors living alone with household market income ≥ S$1,000 | **0.6%** | 94.7% | Expected to be low (no earned income) |
| Seniors living alone in 1–3 room flats | 65.6% | 19.8% | 47.6% (all 1-person households) |
| Seniors with spouse and child in 4-room or larger | 87.4% | 80.8% | 85.4% (all 3+-person households) |
| Home-type mix gap vs reference (TVD): living alone | 0.180 | 0.293 | — |
| Home-type mix gap vs reference (TVD): with family | 0.035 | 0.049 | — |

- Before step 14, 95% of retired seniors living alone got S$1,000 or more of market income, the same mix as working households. With no earned income, most are expected to be low (their income is only savings, rent, pensions, CPF payouts or family allowances). Now it's 0.6%, all from the Tanglin fallback.
- **Note on the income table:** SingStat's table is household **market income** (work income plus investments, rent, pensions, CPF payouts and allowances from family, before government transfers), not income from work only. So some retired seniors living alone really do have S$1,000 or more; placing all of them below S$1,000 is a simplification.
- Both home-type mixes move closer to the national reference.
- Seniors living alone now overshoot the small-flat share (65.6% vs 47.6%). The reference covers 1-person households of **all** ages, including young singles in condos, so seniors living alone may well be more concentrated in small flats than that. There's no age-specific table to confirm.

## 7. Reproducibility (6 towns)

| Check | Result |
| --- | --- |
| Same town, size and seed give identical personas | 6/6 |
| A different seed gives different personas | 6/6 |
| Growing the panel from 30 to 100 keeps the first 30 personas | 6/6 |

---

## What E1 suggests

**Strengths:**
- Personas reproduce each town's census mix within a fraction of a point, for home type, income, age and sex.
- Senior work status and living arrangement match the national tables within 1 point.
- Step 14 gives retired seniors living alone the low income expected without earnings, instead of the working-household mix that 95% of them got before.
- Runs are fully reproducible.

**Weaknesses and limits:**
- **Panel size:** a 30-persona panel is about 14% off the census mix by chance. Results from one run should be read with that in mind (see E4g).
- **No town-level joint table:** home type × income can't be validated independently per town.
- **National senior tables:** labour force and living arrangement are national rates applied to every town. Town differences (e.g. more seniors living alone in older estates) aren't captured.
- **Known exceptions:** Tanglin and Bukit Timah lowest income bands, as above.
- **Not modelled:** income is not linked to age except for retired seniors living alone. Ethnicity, occupation and household size are not drawn.
- **Scope of E1:** it shows that the personas' profiles are realistic, not that their answers are. That's E4.

## Note found while building the check
Using the **same seed in different towns** reuses the same random numbers for labour force and
living arrangement (one stream per step). That's correct for reproducibility within a town, but
results pooled across towns must use a different seed per town, which the script does. The app
itself is unaffected, since each run is one town.
