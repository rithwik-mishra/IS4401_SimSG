# SimSG

Singapore micro-SME AI focus group simulation platform. SimSG builds synthetic
residents (personas) from Singapore census / household data, and has each one
answer a market-research question through an LLM.

All commands below are run from the `backend/` folder.

## Setup

```
cd backend
pip install -r requirements.txt
copy .env.example .env      # then fill in your API key
```

`.env` settings:

| Variable | Purpose |
| --- | --- |
| `SIMSG_PROVIDER` | `openai` or `soclaas` |
| `OPENAI_API_KEY`, `SIMSG_MODEL` | used when the provider is `openai` (default model `gpt-4.1-mini`) |
| `SOCLAAS_BASE_URL`, `SOCLAAS_API_KEY`, `SOCLAAS_MODEL` | used when the provider is `soclaas` (NUS SoC endpoint) |

## Project layout

```
backend/
  app/        FastAPI app (main.py, routers/, services/, schemas.py)
  pipeline/   clean_data.py, fit.py  - turn raw spreadsheets into sampling tables
  agents/     sampler.py, respondent.py  - persona sampling and LLM respondents
  scripts/    Layer 1 / Layer 2 (RAG) data build scripts
  data/       sheets/{raw,cleaned,processed,checks}, chroma_db/, runs/, rag_corpus/
  tests/      pytest tests
```

## Persona pipeline

Run in order:

```
python pipeline/clean_data.py     # data/sheets/cleaned/*.xlsx -> data/sheets/processed/*.csv
python pipeline/fit.py            # iterative proportional fitting of dwelling x income per planning area
```

Then draw personas:

```
python agents/sampler.py "Ang Mo Kio" 10            # 10 personas, random seed
python agents/sampler.py "Ang Mo Kio" 10 --seed 42  # reproducible
python agents/sampler.py --check "Ang Mo Kio"       # draw 20,000 and compare against source tables
```

Sampling order: dwelling + income band, then sex + age group, then ethnic group,
then spending per household member (looked up from HES 2023).

## Respondent agents

Each persona answers one question via the configured LLM provider:

```
python -m agents.respondent "Ang Mo Kio" 5 --dry-run                     # print prompts only, no API call
python -m agents.respondent "Ang Mo Kio" 5 --seed 42                     # 5 real answers
python -m agents.respondent "Ang Mo Kio" 5 --seed 42 --provider soclaas  # use the SoC endpoint
python -m agents.respondent "Tampines" 30 --question "A new bubble tea shop near your home sells a large milk tea for S$6.50. Would you buy one?"
```

Results are saved to `data/runs/<time>_<area>_<seed>.csv` (one row per persona:
profile, exact prompt, answer, tokens used).

## Layer 1 / Layer 2 data scripts

```
python scripts/extract_layer1_seed.py         # layer1_master.csv -> layer1_seed.json
python scripts/scrape_layer2_competitors.py   # OSM Overpass -> raw_osm_competitors.json
python scripts/build_layer2_corpus.py         # -> rag_corpus/layer2_corpus.json
python scripts/ingest_layer2_chroma.py        # -> ChromaDB collection "simsg_layer2_rag"
```

## API

Start the server and open the Swagger docs at http://127.0.0.1:8000/docs:

```
python -m uvicorn app.main:app --reload
```

Test routes: `/api/test/diagnostics` and `/api/test/simulation-context`.

## Tests

```
python -m pytest tests/test_simulation_context_route.py
```
