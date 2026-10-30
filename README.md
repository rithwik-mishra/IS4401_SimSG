# SimSG

Singapore micro-SME AI focus group simulation platform. SimSG builds synthetic
residents (personas) from Singapore census / household data, grounds them in local
neighborhood market reality via ChromaDB Layer 2 qualitative RAG, and has each persona
answer a market-research question through an LLM.

All commands below are run from the `backend/` folder.

## Setup

```
cd backend
python -m venv venv
venv\Scripts\activate       # PowerShell/cmd; on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # then fill in your API key
```

Remember to activate the virtual environment again in each new terminal before running the commands below.

`.env` settings:

| Variable | Purpose |
| --- | --- |
| `SIMSG_PROVIDER` | `openai` or `soclaas` |
| `OPENAI_API_KEY`, `SIMSG_MODEL` | used when the provider is `openai` (default model `gpt-4.1-mini`) |
| `SOCLAAS_BASE_URL`, `SOCLAAS_API_KEY`, `SOCLAAS_MODEL` | used when the provider is `soclaas` (NUS SoC endpoint) |

## System Architecture

```
Layer 1: Census Data & Personas (clean_data.py + fit.py -> sampler.py)
   ├── 29 Singapore Planning Areas (dwelling, income, age, sex, ethnic group, HES 2023 spend)
   └── Statistically representative individual persona panels
          ↓
Layer 2: Local Market Knowledge (ChromaDB collection: simsg_layer2_rag)
   ├── Pricing Benchmarks (hawker kopi vs cafe sets vs bubble tea baselines)
   ├── Competitors & Supply (OSM operating F&B, anchor malls, food courts)
   └── Foot Traffic & Lifestyle (peak hours, retiree density, MRT interchanges)
          ↓
Focus Group Simulation Engine (respondent.py & FastAPI /api/v1/simulation)
   ├── Grounded Prompts combining Persona Profile + Local Neighborhood Context
   └── Structured Answers (would_buy, max_price, likelihood, driver, objection)
```

## Project layout

```
backend/
  app/
    main.py                   - FastAPI application entry point
    schemas.py                - Pydantic models for demographics, RAG, and simulation
    routers/
      simulation.py           - Production routes (/preview, /run, /areas)
      test_simulation.py      - Diagnostics & context inspection routes
    services/
      simulation_context.py   - ChromaDB RAG vector query & context singleton
  agents/
    sampler.py                - Persona sampling from SingStat tables
    respondent.py             - Focus group respondent agent with ChromaDB RAG grounding
  pipeline/
    clean_data.py             - Raw spreadsheets -> tidy processed CSVs
    fit.py                    - Iterative proportional fitting (dwelling x income)
  scripts/
    scrape_layer2_competitors.py - Overpass OSM competitor scraper
    build_layer2_corpus.py       - Qualitative knowledge corpus builder
    ingest_layer2_chroma.py      - ChromaDB vector store ingestion
  data/
    sheets/                   - Official SingStat data (raw, cleaned, processed)
    chroma_db/                - Persistent ChromaDB vector index
    runs/                     - Saved simulation results CSVs
  tests/                      - Automated end-to-end test suites
```

## Persona pipeline

Run in order to rebuild sampling tables from source workbooks:

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

## Respondent agents (with ChromaDB RAG)

Each persona evaluates a concept grounded in both their household profile and the local neighborhood context (pricing norms, competitors, lifestyle):

```
python -m agents.respondent "Ang Mo Kio" 5 --dry-run                     # print prompts with ChromaDB context, no API call
python -m agents.respondent "Ang Mo Kio" 5 --dry-run --no-rag           # ablation: print prompts without ChromaDB context
python -m agents.respondent "Ang Mo Kio" 5 --seed 42                    # 5 real answers with ChromaDB context
python -m agents.respondent "Ang Mo Kio" 5 --seed 42 --provider soclaas # use the SoC endpoint
python -m agents.respondent "Tampines" 30 --question "A new bubble tea shop near your home sells a large milk tea for S$6.50. Would you buy one?"
```

Results are saved to `data/runs/<time>_<area>_<seed>.csv` (one row per persona:
profile, exact prompt, answer, tokens used, RAG metadata).

## Layer 2 Knowledge & Vector Store

```
python scripts/scrape_layer2_competitors.py   # OSM Overpass -> raw_osm_competitors.json
python scripts/build_layer2_corpus.py         # -> rag_corpus/layer2_corpus.json
python scripts/ingest_layer2_chroma.py        # -> ChromaDB collection "simsg_layer2_rag"
```

## API

Start the server and open the Swagger docs at http://127.0.0.1:8000/docs:

```
python -m uvicorn app.main:app --reload
```

Key endpoints:
- `GET /api/v1/simulation/areas`: Supported planning areas & ChromaDB coverage.
- `POST /api/v1/simulation/preview`: Zero-cost preview of personas and grounded prompts.
- `POST /api/v1/simulation/run`: Execute full LLM focus group simulation.
- `GET /api/test/diagnostics`: System diagnostics (ChromaDB health, doc count).

## Tests

Run all automated unit and API integration tests:

```
python -m pytest tests
```
