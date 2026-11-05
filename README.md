# SimSG

Singapore micro-SME AI focus group simulation platform. SimSG builds synthetic
residents (personas) from Singapore census / household data, grounds them in local
neighborhood market reality via ChromaDB Layer 2 qualitative RAG, and has each persona answer a market-research question through an LLM.

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
   ├── 29 Singapore Planning Areas (dwelling, income, age, sex, HES 2023 eating-out spend)
   └── Statistically representative individual persona panels
          ↓
Layer 2: Local Market Knowledge (ChromaDB collection: simsg_layer2_rag)
   ├── Pricing Benchmarks (hawker kopi vs cafe sets vs bubble tea baselines)
   ├── Competitors & Supply (OSM operating F&B, anchor malls, food courts)
   └── Foot Traffic & Lifestyle (peak hours, retiree density, MRT interchanges)
          ↓
Focus Group Simulation Engine (POST /api/simulate)
   ├── Pitch reader -> concept, price, town
   ├── Persona answers (execution_engine.py): two-part interview per persona
   │     1. own 4 price limits, with the seller's price hidden
   │     2. buy %, reaction and objection once the price is shown
   ├── Price analysis (Van Westendorp PSM) and cohort breakdown
   └── Report writer -> verdict, objections, pivot
```

Each AI step falls back to a rule-based backup when no API key is set or a call fails.

## Project layout

```
backend/
  app/
    main.py                   - FastAPI application entry point
    schemas.py                - Pydantic models for demographics, RAG, and simulation
    routers/
      simulate.py             - Main pipeline (/api/simulate, /api/simulate/preview, /api/ai-status)
      simulation.py           - Planning-area lookup (/api/simulation/areas)
      test_simulation.py      - Diagnostics & context inspection routes
    services/
      sampler.py              - Persona sampling from SingStat census tables
      simulation_context.py   - ChromaDB RAG vector query & context singleton
      query_decomposer.py     - Pitch reader
      econometrics.py         - Van Westendorp PSM and cohort statistics
      llm_settings.py         - AI provider and model from .env
    agents/
      execution_engine.py     - Persona answers (AI, with rule-based backup)
      synthesis_agent.py      - Report writer (AI, with rule-based backup)
    legacy/                   - Replaced code kept for reference (not used by the app)
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
python -m app.services.sampler "Ang Mo Kio" 10            # 10 personas, random seed
python -m app.services.sampler "Ang Mo Kio" 10 --seed 42  # reproducible
```

Sampling order: dwelling + income band, then sex + age group (drawn separately),
then an exact age and income within the drawn group and band, then the HES 2023
eating-out spend per household member for the dwelling type. Income is shown to
the AI and in the app only as its census band (household income from work).

## Legacy respondent agent

The earlier respondent pipeline (`respondent.py`, its sampler and the
`/simulation/preview` and `/simulation/run` routes) was replaced by `/api/simulate`
and kept in `app/legacy/`. It still runs by hand:

```
python -m app.legacy.respondent "Ang Mo Kio" 5 --dry-run    # print prompts with ChromaDB context, no API call
python -m app.legacy.integrated_simulation_checks           # confirm the legacy code still works
```

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

The frontend's Testing panel (beaker icon, top right) shows the AI model in use, which
steps of the last run used the AI or the backup, and a seed box for repeatable runs.

Key endpoints:
- `POST /api/simulate`: Run the full focus group (pitch -> personas -> answers -> price analysis -> report).
- `POST /api/simulate/preview`: Zero-cost preview of the personas and the exact prompts they would be sent.
- `GET /api/ai-status`: Which AI provider and model runs would use (or the rule-based backup).
- `GET /api/simulation/areas`: Supported planning areas & ChromaDB coverage.
- `GET /api/test/diagnostics`: System diagnostics (census areas, ChromaDB health, doc count).

## Tests

Run all automated unit and API integration tests:

```
python -m pytest tests
```
