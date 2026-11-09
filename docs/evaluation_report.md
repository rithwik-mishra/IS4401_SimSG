# 6. Evaluation

## 6.1 Evaluation approach

SimSG produces a "silicon sample": a large language model answers as a panel of synthetic
residents. Earlier studies show that such samples fail in predictable ways:
- averages can look right while the relationships between variables are wrong (Bisbee et al., 2024)
- answers vary less than real people's (Wang et al., 2025)
- models anchor on numbers given in the prompt (Huang et al., 2025)
- stated willingness to pay is sometimes wrong (Brand, Israeli & Ngwe, 2023)

For this reason, researchers recommend these tools as a fast, low-cost pre-test before real customer
research, not as a replacement for it (Sarstedt et al., 2024). We evaluate SimSG on that basis.

SimSG is a pipeline: it reads the pitch, draws personas, interviews them, analyses prices and
writes a report. We test each stage with the method that suits it, and ask three questions:

1. **Faithful:** where ground truth exists (census tables, hand-labelled pitches, hand-computed price
   curves), does the system reproduce it?
2. **Sensible:** where no ground truth exists, do answers move in the expected direction? For
   example, does demand fall as price rises, and do higher-income personas accept higher prices?
3. **Stable:** do repeated runs give similar results?

**Table 6.1. Evaluation overview**

| Section | Stage tested | Question | Method | Evidence | Ground truth / baseline |
| --- | --- | --- | --- | --- | --- |
| 6.2 Persona realism | Persona sampler | Do personas match the residents of the chosen town? | 40,000 personas per planning area compared with census tables; 500 panels of 30 personas; senior household rule tested with and without | Metrics: largest gap, TVD, SRMSE | SingStat census tables; the sampler without the senior household rule |
| 6.3 Pitch understanding | Pitch reader | Are the price, town and category read correctly? | 24 hand-labelled pitches (12 typical, 12 challenging), read by the AI and by the rule-based backup | Test cases: accuracy per field | Hand labels; rule-based backup |
| 6.4 Response behaviour | Persona interviews | Are answers coherent, sensible, free of price anchoring and stable? | One pitch run 10 times: a price sweep (S$3–12) with the same personas, repeat runs with the same and with different personas; price-limit ordering; income comparison | Metrics and directional checks | Expected direction (demand falls as price rises; higher income accepts higher prices) |
| 6.5 Price analysis | Van Westendorp analysis | Are the price points calculated correctly? | A hand-calculated worked example plus property and edge-case unit tests | Test cases | Hand calculation |

Together these cover three kinds of evidence:
- **Test cases:** 6.3 and 6.5
- **Quantitative metrics:** 6.2 and 6.4
- **Baselines:** the rule-based backup (6.3), and the sampler with and without the senior household
  rule (6.2)

Each section gives a representative example and at least one challenging case, for example:
- towns whose census data cannot satisfy the senior household rule (6.2)
- pitches with prices in words or cents, spelling mistakes and unsupported towns (6.3)
- the S$7 artisan kopi pitch at prices from S$3 to S$12 (6.4)

Section 6.6 brings the results together into the system's strengths and weaknesses.

## 6.2 Persona realism

**What we tested.** Every SimSG answer comes from a synthetic persona, so the results are only
meaningful if the panel resembles the people who live in the chosen town. We tested whether the
persona sampler reproduces each town's census profile (home type, household income, age and sex),
whether its senior household rule produces believable households, and how much a 30-person panel
varies by chance.

**Method.** Ground truth is the SingStat census tables in `backend/data/sheets/processed/`. These
give residents by age and sex, and households by home type and by monthly household market
income for each planning area. Market income is income from work plus investments, rent,
pensions, CPF payouts and allowances from family, before government transfers and tax. The data also includes the 2025 national tables for senior labour force and living arrangements. The
script `backend/scripts/evaluate_sampler.py` (seed 2026) runs four main checks:

1. **Census match:** draw 40,000 personas for each of the six towns in the app and compare their
   mix with the town's census tables. The same check was also run on all 29 planning areas with
   census data.
2. **Senior household rule:** for 290,000 personas pooled across all areas, give seniors a home
   and income in two ways: with SimSG's rule (chosen to fit the person's work status and living
   arrangement), and without it. Compare the incomes and home types that seniors receive.
3. **Panel size:** draw 500 panels of 30 personas (the size used in the app) in each of the six
   towns, and measure how far each panel is from the census.
4. **Supporting checks:** the fitted home type × income tables against published totals; senior
   work and living rates against national tables; and reproducibility with the same seed.

Three measures compare persona shares with census shares (Table 6.2).

**Table 6.2. Measures used**

| Measure | Meaning | Perfect | Rough guide |
| --- | --- | --- | --- |
| Largest gap | The biggest difference, in percentage points, in any single group (+ too many personas, − too few) | 0 | Under 1 pt excellent; 1–3 pts good |
| Total variation distance (TVD) | The share of personas that would need to change group to match the census | 0 | Under 0.01 excellent; 0.01–0.05 good |
| Standardised root mean square error (SRMSE) | Typical error per group ÷ average group share; the standard fit measure in population synthesis | 0 | Under 0.05 excellent |

### 6.2.1 Personas match the census of each town

**Table 6.3. Persona mix vs census, six app towns (40,000 personas each)**

| Town | Home type: largest gap | Home type TVD | Income: largest gap | Income TVD | Age × sex TVD |
| --- | --- | --- | --- | --- | --- |
| Ang Mo Kio | −0.57 pts (3 room) | 0.010 | +0.22 pts (S$35,000+) | 0.008 | 0.013 |
| Bukit Merah | −0.40 pts (4 room) | 0.007 | +0.22 pts (below S$1,000) | 0.008 | 0.012 |
| Bukit Timah | −0.44 pts (landed) | 0.005 | **+1.69 pts (below S$1,000)** | 0.018 | 0.011 |
| Jurong East | +0.32 pts (1-2 room) | 0.004 | −0.17 pts (S$25,000–29,999) | 0.007 | 0.010 |
| Outram | −0.68 pts (3 room) | 0.009 | −0.42 pts (S$3,000–3,999) | 0.012 | 0.011 |
| Tampines | +0.29 pts (1-2 room) | 0.004 | −0.13 pts (S$30,000–34,999) | 0.007 | 0.011 |

*Representative result.* In a typical town such as Tampines, no home type is more than 0.3 points
from the census and about 1 persona in 250 is in the wrong home type (TVD 0.004). Every town's
largest age × sex gap is under 0.25 points, and the male/female split is within 0.35 points. Across
all 29 planning areas, the median largest gap is 0.35 points for home type and 0.20 points for
income.

*Challenging case: Bukit Timah.* The lowest income band (below S$1,000) is 1.69 points too high.
National rates predict that about 2.7% of Bukit Timah's personas are retired seniors living alone,
and the senior household rule places them in the lowest income band. But the census shows only
1.1% of the town's households in that band, so the band overflows. A similar case outside the app
is Tanglin, which has no households below S$1,000 at all, so these seniors fall back to the lowest
band that exists (+2.39 points). Both cases come from a conflict between national senior rates and
small local income bands, not from random error.

### 6.2.2 Effect of the senior household rule

For seniors aged 65 and over, SimSG chooses home type and income to fit the person: retired seniors
living alone are placed in the lowest market-income band, and seniors living alone or with a spouse
and child are steered towards home types where households of that size are common. To measure what
this rule does, we drew the same personas with and without it (an ablation).

**Table 6.4. Senior households with and without the senior household rule**

| Measure | With rule | Without rule |
| --- | --- | --- |
| Retired seniors living alone with household market income ≥ S$1,000 | **0.6%** | 94.7% |
| Seniors living alone in 1–3 room flats | 65.6% | 19.8% |
| Seniors living with spouse and child in 4-room or larger homes | 87.4% | 80.8% |

Without the rule, 95% of retired seniors living alone were given S$1,000 or more of market income,
the same mix as working households, even though no one in their household works. With the rule,
0.6% were, all from Tanglin, which has no households below S$1,000 in its census table. The rule
also moves seniors living alone into smaller flats (19.8% to 65.6%) and seniors living with family
into bigger homes. These changes make each senior's household consistent with the rest of their
profile, and they do not disturb each town's overall census match (Section 6.2.1), because the
rule moves other personas out of the lowest income band to compensate.

This comparison shows what the rule does, not that the resulting figures are accurate. No
published table in our data gives home type or market income by age and living arrangement.
Because market income includes rent, investments, pensions and allowances from family, some
retired seniors living alone do receive S$1,000 or more, so placing all of them in the lowest band
is a simplification. The rule may also over-concentrate seniors living alone in small flats,
because the lowest income band is made up mostly of small flats.

### 6.2.3 A 30-person panel varies by chance

**Table 6.5. How far one 30-persona panel is from the census (500 panels × 6 towns)**

| Characteristic | Mean TVD | 90th percentile TVD |
| --- | --- | --- |
| Home type (6 groups) | 0.137 | 0.205 |
| Household income (6 groups) | 0.150 | 0.221 |
| Age (5 groups) | 0.141 | 0.213 |

In the app, each simulation uses a panel of 30 personas. Even though the sampler matches the census
closely, a panel this small cannot reflect the town's mix exactly, just as 30 randomly chosen real
residents would not. Across 500 random panels per town, a typical panel had about 14 in every 100
personas in a different home type, income group or age group from what the census would predict,
and 1 panel in 10 was more than 20 in 100 off. This chance variation is more than ten times larger
than the sampler's own error (fewer than 1 in 100 personas in Table 6.3). How representative a
single run is therefore depends mainly on the panel size, not on the sampler, and results from one
run should be treated as estimates. Section 6.4 tests how much the final results change between
runs.

### 6.2.4 Supporting checks

- **Census fitting:** the fitted home type × income table for every town adds up exactly to the
  town's published totals (gap 0.00 points). Added up across all towns, the income mix within each
  home type stays within 2.3 points of the national table, with landed homes drifting the most.
- **Seniors:** pooled across all areas, persona labour-force rates (by age and sex) are within
  0.9 points of the 2025 national rates (median 0.43 points), and living arrangements are within
  0.9 points (median 0.21 points).
- **Reproducibility:** in all six app towns, the same seed gives identical personas, a different
  seed gives different personas, and growing a panel from 30 to 100 keeps the first 30 personas.

### 6.2.5 What the results suggest

**Strengths.** The sampler reproduces each town's census mix with about 1 persona in 100 or fewer
in the wrong group, and fewer than 1 in 50 in every case. Senior work and living patterns match national
rates within 1 point, the senior household rule makes seniors' homes and incomes consistent with their
profile without disturbing the census match, and every run can be reproduced.

**Weaknesses.**
- **Panel size.** One 30-person run is about 14% off the census mix by chance, so results from a
  single run should be treated as estimates and compared across several runs (see 6.4).
- **Conflicting data sources.** National senior rates do not always fit local income bands
  (Bukit Timah, Tanglin).
- **All-or-nothing senior income rule.** Every retired senior living alone is placed below
  S$1,000 of market income, although some have rent, investment income or allowances from family
  above that. This likely understates the income of better-off retirees.
- **Untested combinations.** No town-level home type × income table is published, so that
  combination can be checked only against the national pattern. Income is linked to age only for
  retired seniors living alone, and ethnicity, occupation and household size are not modelled.
- **Partly circular.** The sampler is built from the same census tables it is compared with. A
  close match shows that the sampler works as designed, but cannot reveal errors or gaps in the
  census itself. Persona *profiles* are realistic, but this does not show that their *answers*
  are; that is tested in 6.4.

## 6.3 Pitch understanding

**What we tested.** Every simulation starts by reading the owner's free-text pitch into three
fields: the town, the price and the business category. A mistake here affects everything after it:
the wrong town gives the wrong residents and local context, and the wrong price changes every
purchase decision. We tested how accurately the pitch reader extracts these fields, using the AI
model (gpt-4o-mini) and the rule-based backup that runs when the AI is unavailable.

**Method.** We wrote 24 test pitches and labelled the correct answer for each field by hand
(`backend/data/evaluation/e2/pitches.csv`): 12 typical pitches and 12 challenging ones, such as
prices written in words or cents, spelling mistakes and informal English. For pitches with no
price or a town outside the six supported, the correct answer is the app's designed default
(S$4.50 and Ang Mo Kio). For a price range, any price within the range counts as correct. Both
readers processed every pitch (`backend/scripts/evaluate_pitch_reader.py`), and we report the share
of pitches with each field correct.

**Table 6.6. Pitch reader accuracy (% of pitches correct)**

| Reader | Pitches | Town | Price | Category | All three fields |
| --- | --- | --- | --- | --- | --- |
| AI | 12 typical | 100% | 100% | 100% | 100% |
| AI | 12 challenging | 92% | 100% | 100% | 92% |
| Rule-based backup | 12 typical | 100% | 100% | 83% | 83% |
| Rule-based backup | 12 challenging | 92% | 83% | 83% | 67% |
| **AI** | **All 24** | **96%** | **100%** | **100%** | **96%** |
| **Rule-based backup** | **All 24** | **96%** | **92%** | **83%** | **75%** |

*Representative example.* "S$7 artisanal single-origin kopi bar in Ang Mo Kio with specialty cold
brew and toast sets" is read correctly by both: Ang Mo Kio, S$7.00, beverage.

**Table 6.7. Challenging cases where the readers differ**

| Pitch (shortened) | Correct | AI | Rule-based backup |
| --- | --- | --- | --- |
| "kueh at 80 cents each" | S$0.80, bakery | S$0.80, bakery | **S$80.00**, beverage |
| "chicken rice stall… five bucks a plate" | S$5.00 | S$5.00 | S$4.50 (default; words not read) |
| "Bubbel tea shop in Tampenis" | Tampines | Tampines | Ang Mo Kio (typo not matched) |
| "wanna open a lor mee stall at tampines" | dining | dining | retail |
| "salted egg lava bao kiosk" | bakery | bakery | beverage (no matching keyword) |
| "Brunch restaurant in Tiong Bahru" | Bukit Merah | **Ang Mo Kio** | Bukit Merah |

The AI handled prices written in words or cents, spelling mistakes and informal English. Its one
error was the neighbourhood name Tiong Bahru, which lies in Bukit Merah: the AI did not link the
two and the app fell back to its default town. The rule-based backup recognises Tiong Bahru because
it is in its list of town aliases, but it misses anything outside its keyword lists. Its most
serious error was reading "80 cents" as S$80, a price 100 times too high that would make every
persona refuse to buy. A pitch in an unsupported town ("Craft beer bar in Bedok") is quietly
mapped to Ang Mo Kio by both readers, as designed, but the owner is not told.

**What it means.** With the AI, the pitch reader is reliable: all fields were correct in 23 of 24
pitches, including the challenging ones. The rule-based backup is adequate for clearly written
pitches but fails on informal wording, and its errors can be severe, so results produced without
the AI should be treated with more caution. Weaknesses are the silent fallback to Ang Mo Kio for
unknown or unrecognised towns, which could produce a simulation for the wrong place, and the small
test set: 24 pitches written by the team, each run once, cannot cover every way owners describe
their business.

## 6.4 Response behaviour

**What we tested.** The persona answers have no "correct" value, because no real survey of these
residents exists. Instead, we tested whether the AI personas behave in ways that make sense
(Section 6.1, questions 2 and 3): do they buy less when the price rises, do richer personas
accept higher prices, are their answers internally consistent, are their price limits free of
anchoring on the seller's price, and do repeated runs give similar results?

**Method.** We ran one pitch, "Artisanal single-origin kopi bar in Ang Mo Kio with specialty cold
brew and toast sets", through the full app ten times, with 30 personas per run and the AI used for
every persona (`backend/scripts/evaluate_responses.py`; saved responses in
`backend/data/evaluation/e4/`):
- **Price sweep:** S$3, 5, 7, 9 and 12, with the same 30 personas (seed 2026), so only the price
  changes.
- **Repeat runs:** three more runs at S$7 with the same personas, so only the AI's own randomness
  changes, and two runs at S$7 with different personas (seeds 2027 and 2028).

Each persona is interviewed in two parts: first it gives its four price limits without seeing the
seller's price, then it is shown the price and gives its purchase likelihood (0–100%).

### 6.4.1 Demand falls as price rises

**Table 6.8. Price sweep with the same 30 personas**

| Seller's price | Mean purchase likelihood | Personas at 50% or more | Optimal price point (OPP) |
| --- | --- | --- | --- |
| S$3 | 77.7% | 30 of 30 | S$5.05 |
| S$5 | 66.8% | 29 of 30 | S$4.47 |
| S$7 | 39.7% | 11 of 30 | S$4.47 |
| S$9 | 16.3% | 0 of 30 | S$4.32 |
| S$12 | 10.7% | 0 of 30 | S$5.34 |

Purchase likelihood falls at every step, from 77.7% at S$3 to 10.7% at S$12, giving a
downward-sloping demand curve. The steepest drop is between S$5 and S$7, around the price where
most personas start calling the product "expensive" (median S$6). The optimal price point stays
between S$4.32 and S$5.34 whatever price is shown, as intended, because personas give their
price limits before seeing the seller's price.

### 6.4.2 Answers are internally consistent

- **Price limits in order:** in all 300 answers, "too cheap" ≤ "cheap" ≤ "expensive" ≤ "too
  expensive" (100%).
- **Purchase matches price limits:** in 45 answers the seller's price was above the persona's own
  "too expensive" limit; in 43 of them (96%) the persona gave a purchase likelihood of 20% or less.

### 6.4.3 Higher-income personas accept higher prices

**Table 6.9. Price limits by household market income (four S$7 runs with the same personas)**

| Household income | Answers | Median "expensive" | Median "too expensive" | Mean purchase likelihood |
| --- | --- | --- | --- | --- |
| Below S$3,000 | 24 | S$5.00 | S$8.00 | 19.2% |
| S$3,000–9,999 | 20 | S$5.75 | S$8.00 | 32.0% |
| S$10,000 and above | 76 | S$7.00 | S$10.00 | 47.9% |

Price limits and purchase likelihood rise with household income, as expected. The two lower groups
are small (24 and 20 answers from 6 and 5 personas), so the differences between them are less
certain than the gap to the highest group.

### 6.4.4 No anchoring on the seller's price

Because the price is hidden in the first part of the interview, the personas' price limits should
not follow the seller's price. Across the sweep, the median "cheap" limit stayed at S$4.00 and the
median "too expensive" limit at S$10.00 at every price, and the median "expensive" limit moved only
between S$6.00 and S$6.50. In 13 of 150 answers the "expensive" or "too expensive" limit equalled
the seller's price, but this is no more often than chance: S$7 was a common answer whatever price
was shown (11 personas when shown S$3, 11 when shown S$12, but only 6 when shown S$7 itself).

### 6.4.5 Results are stable across repeat runs

**Table 6.10. Repeat runs at S$7**

| Runs | Mean purchase likelihood | Range | Standard deviation | OPP range |
| --- | --- | --- | --- | --- |
| Same 30 personas, 4 runs | 39.5% | 37.7–40.7% | 1.3 points | S$3.89–5.05 |
| Different personas, 3 runs | 36.8% | 34.0–39.7% | 2.8 points | S$3.89–4.47 |

With the same personas, the AI's randomness moves mean purchase likelihood by only about 1 point.
A new panel of personas moves it by about 3 points, consistent with the panel-size effect in
6.2.3. The optimal price point is less stable: it varied by up to S$1.16 across runs, because with
30 personas the price curves are jagged and small changes in a few answers shift where they cross.

### 6.4.6 What the results suggest

**Strengths.** The AI personas behave like sensible consumers: demand falls as price rises, richer
personas accept higher prices, answers are consistent with each other, and the two-part interview
keeps price limits independent of the seller's price. The overall purchase likelihood is stable to
within a few points across runs.

**Weaknesses.**
- **Sensible is not the same as accurate.** These tests show the answers move in the right
  direction, not that the levels are right; for example, whether 40% of Ang Mo Kio residents would
  really consider a S$7 artisan kopi. Checking that would need a survey of real residents.
- **Price points vary.** The optimal price point can move by more than S$1 between runs, so a single
  run's price recommendation should be treated as a range.
- **Narrow test.** Only one pitch, in one town, was tested; other categories and towns may behave
  differently. Rewording the same pitch, and comparison with the earlier single-question prompt,
  were not tested in the time available.

## 6.5 Price analysis

SimSG turns the personas' four price answers into Van Westendorp price curves and four price points,
including the optimal price point it recommends. This step uses no AI, so it was checked with unit
tests (`backend/tests/test_econometrics.py`). In a worked example of four respondents, the price
points calculated by hand (S$5.00, S$5.50, S$6.00 and S$6.50) match the code's output exactly, and a
second hand-checked example confirms the average purchase likelihood and net sentiment. Further
tests confirm that the curves slope the right way and the price points fall in order on 40
generated respondents, and that empty, single-respondent and maximum-price-only inputs are handled
without failing; all 12 tests pass.

Two design choices affect how results should be read. Price limits given out of order are sorted
silently rather than flagged, although none of the 300 AI answers in Section 6.4 needed this. With
30 personas the curves are jagged, so a single run's price points should be read together with the
stability results in Section 6.4.

## 6.6 Strengths and weaknesses

**Table 6.11. Summary of results**

| Section | Main result | Verdict |
| --- | --- | --- |
| 6.2 Persona realism | Personas match each town's census with about 1 in 100 or fewer in the wrong group; senior work and living rates within 1 point of national figures | Strength |
| 6.2 Panel size | A 30-person panel is about 14% off the census mix by chance | Weakness |
| 6.3 Pitch understanding | AI reads town, price and category correctly in 23 of 24 pitches; rule-based backup in 18 of 24 | Strength (AI); weakness (backup) |
| 6.4 Response behaviour | Demand falls with price, richer personas accept higher prices, answers 100% in order, no anchoring; purchase likelihood stable within about 3 points | Strength |
| 6.4 Price points | The optimal price point moves by up to S$1.16 between runs | Weakness |
| 6.5 Price analysis | Matches a hand calculation exactly; all 12 tests pass | Strength |

**Strengths.**
1. **Grounded personas.** Each panel is drawn from SingStat 2025 tables for the chosen town, and
   the sampler reproduces those tables closely. The AI's role is limited to reading the pitch,
   answering as each persona and writing the report; who the personas are, and the price
   calculations, are fixed by data and code.
2. **Sensible behaviour.** The AI personas respond to price and income in the expected directions,
   and the two-part interview prevents them from simply repeating the seller's price, a known
   weakness of AI survey respondents.
3. **Reliable input handling.** With the AI, the pitch reader copes with informal wording, prices in
   words or cents and spelling mistakes.
4. **Reproducible and checkable.** A run can be repeated exactly from its seed, the price analysis
   is verified against a hand calculation, and every evaluation in this chapter can be rerun with
   the scripts in `backend/scripts/`.

Besides the evaluation scripts, the backend has 173 automated tests, all passing, which can be rerun
with one command (`pytest`) after any change (Table 6.12). Three groups directly support results in
this chapter: the interview prompt tests confirm that the seller's price is hidden in the first part
of every interview (6.4.4); the fallback tests confirm that the rule-based backup runs when the AI is
unavailable (6.3); and the reproducibility tests confirm that a seed always gives the same personas
(6.2). The remaining tests cover configuration and security, which matter for running the app safely
but do not measure its effectiveness.

**Table 6.12. Automated tests in the backend**

| Test file | What it checks | Tests |
| --- | --- | --- |
| `test_deliverable_services.py` | Pitch reader rules; persona sampler matches census tables, keeps ages and incomes within their bands, applies the senior rules and is reproducible | 60 |
| `test_agents.py` | Interview prompts (price hidden in part 1, no reference prices, market-income wording); two-part interview and its fallback; report writer fallback; a full 30-persona run | 37 |
| `test_llm_settings.py` | Which AI provider and model each step uses; reporting which steps used the AI or the backup | 21 |
| `test_simulation_context_route.py` | Local context retrieval by town; password protection on admin routes | 16 |
| `test_econometrics.py` | Price curves and price points, summary statistics, customer groups, edge cases (6.5) | 12 |
| `test_simulate_preview.py` | The preview shows the same personas and prompts that a real run sends | 10 |
| `test_seed_reproducibility.py` | The same seed gives an identical result; a different seed gives different personas | 5 |
| `test_simulation_pipeline.py` | A full simulation through the API, including town and price overrides | 4 |
| `test_rate_limiter.py` | Limits on requests per session | 4 |
| `test_security_hardening.py` | Allowed origins, security headers, safe error messages | 4 |
| **Total** | | **173** |

**Weaknesses.**
1. **No comparison with real people.** All tests check consistency and direction, not whether the
   levels are right. We cannot say that real Ang Mo Kio residents would give a 40% purchase
   likelihood for a S$7 artisan kopi. This is the main limitation, and it matches the research
   view that AI respondents suit early pre-testing rather than replacing customer research.
2. **Small panels.** Thirty personas keep each run fast and cheap, but make the panel mix and,
   especially, the optimal price point vary between runs. Results should be read as ranges, ideally
   from several runs.
3. **Simplifying rules.** National senior rates are applied in every town, and every retired senior
   living alone is placed in the lowest income band, which does not fit wealthy towns such as Bukit
   Timah.
4. **Backup quality.** Without the AI, the rule-based pitch reader makes serious errors (such as
   reading 80 cents as S$80), and unrecognised towns quietly fall back to Ang Mo Kio without telling
   the owner.
5. **Limited scope of testing.** Response behaviour was tested on one pitch in one town, and the
   tool was not tested with business owners, so its usability and usefulness for real decisions
   remain unmeasured.

**Future work.** The most valuable next steps are: a small survey of real residents to compare
price limits and purchase likelihood with the personas; running several panels per simulation and
reporting ranges; extending the response tests to other categories and towns; warning the owner when
the pitch names an unsupported town; and a usability study with micro-SME owners.

