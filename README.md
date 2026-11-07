# SimSG: Singapore Micro-SME AI Focus Group Simulation Platform

SimSG is an empirical market-validation platform designed for Singapore micro-SMEs and retail entrepreneurs. It constructs synthetic Singapore resident cohorts (personas) grounded in official SingStat Census and Household Expenditure Survey (HES 2023) tables, enriches them with qualitative neighbourhood dynamics via ChromaDB Layer 2 RAG, and conducts concurrent simulated focus groups using large language models and econometric Price Sensitivity Meter (PSM) modeling.

---

## System Architecture

```
Layer 1: SingStat Census & Demographics (clean_data.py + fit.py -> sampler.py)
   ├── 29 Singapore Planning Areas (dwelling, income, age, sex, HES 2023 dining spend)
   └── Statistically representative Monte Carlo synthetic persona cohorts (N=30)
          ↓
Layer 2: Local Qualitative Knowledge (ChromaDB collection: simsg_layer2_rag)
   ├── Pricing Norms & Benchmarks (hawker kopi vs cafe sets vs bubble tea baselines)
   ├── Competitors & Local Supply (OpenStreetMap F&B nodes, anchor malls, food courts)
   └── Foot Traffic & Neighborhood Lifestyle (peak hours, retiree density, MRT nodes)
          ↓
Focus Group Simulation Engine (POST /api/simulate)
   ├── Step 1: Pitch Reader & Query Decomposer (product concept, price, town)
   ├── Step 2: Layer 2 ChromaDB Semantic Neighborhood Context Retrieval
   ├── Step 3: Seeded Monte Carlo Demographic Resident Sampling
   ├── Step 4: Persona Evaluations (execution_engine.py): Two-part interview per persona
   │     - Part 1: Blind Price Limits (4 Van Westendorp thresholds with price hidden)
   │     - Part 2: Purchase Decision (buy %, sentiment, quote, objection with price shown)
   ├── Step 5: Econometric Engine (NumPy/SciPy Van Westendorp curves: OPP, PMC, IPP, PME)
   ├── Step 6: Macro Synthesis Agent (strategic verdict, objection clusters, operational pivots)
   └── Step 7: Streamlined Response ([recommendations, 5 representative personas, statistics])
```

---

## Prerequisites & Dependencies

Before running SimSG locally, ensure you have the following installed:
- **Python**: Version 3.10 to 3.13
- **Node.js**: Version 18.x or higher (with `npm` 9+)

### Package Specifications
- **Backend Python Dependencies**: Documented in [`backend/requirements.txt`](file:///c:/Users/mishr/Desktop/IS4401_SimSG/backend/requirements.txt) (FastAPI, Uvicorn, Pydantic, ChromaDB, NumPy, Pandas, OpenAI) and development/testing tools in [`backend/requirements-dev.txt`](file:///c:/Users/mishr/Desktop/IS4401_SimSG/backend/requirements-dev.txt) (pytest, openpyxl, httpx).
- **Frontend Node Packages**: Defined in [`frontend/package.json`](file:///c:/Users/mishr/Desktop/IS4401_SimSG/frontend/package.json) (React 19, TypeScript, Vite, Tailwind CSS v4, Motion, Recharts, Radix UI).

---

## Local Setup & Quickstart

SimSG runs as a decoupled full-stack application. You will run the FastAPI backend on port `8000` and the Vite React frontend on port `5173`.

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows (PowerShell/CMD)
# source venv/bin/activate  # On macOS/Linux

# Install required Python dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Optional: for offline data pipelines & pytest

# Initialize environment configuration
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Configure your `.env` file with your credentials (see [Environment Configuration](#environment-configuration) below).

Start the FastAPI application:
```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- Interactive API Documentation (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Frontend Setup

In a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install Node modules
npm install

# Start local development server
npm run dev
```

The frontend will start at [http://localhost:5173](http://localhost:5173). The Vite development server automatically proxies requests matching `/api/*` directly to `http://127.0.0.1:8000`.

---

## Environment Configuration

Configure `backend/.env` according to your deployment model:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SIMSG_PROVIDER` | `openai` | AI provider choice (`openai` or `soclaas`). |
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key for live GPT evaluations. |
| `SIMSG_MODEL` | `gpt-4.1-mini` | OpenAI model identifier used across all AI stages. |
| `SOCLAAS_BASE_URL` | `https://soclaas-api.comp.nus.edu.sg/v1` | Base URL if using the NUS SoCLaaS endpoint. |
| `SOCLAAS_API_KEY` | *(empty)* | SoCLaaS API authentication key. |
| `SOCLAAS_MODEL` | `default` | Model identifier for SoCLaaS. |
| `RATE_LIMIT_PER_HOUR`| `10` | Maximum simulation executions allowed per hour per session ID. |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated list of additional allowed CORS origins. |

---

## Zero-Cost Simulation & Mock Mode

SimSG includes complete offline and zero-cost simulation modes:

1. **Rule-Based Fallback (No API Key Required)**:
   If no API key is provided in `.env`, SimSG automatically switches to deterministic, rule-based consumer behavior models and econometric algorithms. All census tables, ChromaDB passages, and PSM calculations operate completely free with zero external API calls.
2. **Explicit Mock Data Payload**:
   You can send `"use_mock": true` in the `POST /api/simulate` payload to instantly return pre-computed Singapore cafe sample benchmarks.
3. **Frontend Preview Mock Fallback**:
   If the backend is offline during frontend previews, the user interface gracefully falls back to synthetic sample results while notifying the user via a banner. Rate limit rejections (HTTP 429) are never masked.

---

## Security & Access Control

1. **Hourly Rate Limiter**:
   - Simulation executions (`POST /api/simulate` and `/simulate`) are capped by default at **10 requests per hour per session ID**.
   - Sessions are tracked via the `X-Session-ID` header (automatically created and persisted in browser `localStorage`).
   - Rate limit status is returned on every response: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and `Retry-After`.
2. **Protected Testing & Diagnostic Routes (`IS4401`)**:
   - Testing routes exposing raw vector passages or database introspection require authentication:
     - `GET /api/test/diagnostics`
     - `GET /api/test/simulation-context`
     - `POST /api/test/simulation-context`
     - `POST /api/simulate/preview`
   - Supply the password `IS4401` via:
     - Header: `X-Test-Password: IS4401` or `X-Admin-Password: IS4401`
     - Bearer Token: `Authorization: Bearer IS4401`
     - Query Parameter: `?password=IS4401`
3. **CORS & HTTP Security Headers**:
   - CORS is restricted to localhost ports (`5173`, `3000`) and production `simsg.vercel.app` (including authenticated Vercel preview branch domains).
   - Responses enforce `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`.
4. **Data Safety & Prompt Redaction**:
   - Internal system prompts, prompt construction templates, and raw ChromaDB vector chunks are shielded from public API outputs.
   - Global unhandled exception handlers catch backend errors and return sanitized JSON responses without leaking tracebacks, disk paths, or credentials.

---

## Project Layout

```
IS4401_SimSG/
├── backend/
│   ├── app/
│   │   ├── main.py                   - FastAPI application entry point & CORS
│   │   ├── schemas.py                - Pydantic models for demographics, RAG, & simulations
│   │   ├── routers/
│   │   │   ├── simulate.py           - Main focus group pipeline (/api/simulate, /api/ai-status)
│   │   │   ├── simulation.py         - Planning-area lookup (/api/simulation/areas)
│   │   │   └── test_simulation.py    - Password-protected diagnostic routes (/api/test/*)
│   │   ├── services/
│   │   │   ├── rate_limiter.py       - Sliding-window in-memory rate limiter
│   │   │   ├── sampler.py            - Monte Carlo persona sampler from SingStat tables
│   │   │   ├── simulation_context.py - ChromaDB RAG vector query singleton
│   │   │   ├── query_decomposer.py   - Pitch attribute & price extractor
│   │   │   ├── econometrics.py       - Van Westendorp PSM and cohort statistics
│   │   │   └── llm_settings.py       - Model settings & API provider resolution
│   │   ├── agents/
│   │   │   ├── execution_engine.py   - Persona answers (LLM + rule-based backup)
│   │   │   └── synthesis_agent.py    - Report writer (LLM + rule-based backup)
│   │   └── legacy/                   - Reference code and historical benchmarks
│   ├── pipeline/
│   │   ├── clean_data.py             - Raw spreadsheets -> tidy processed CSVs
│   │   └── fit.py                    - Iterative proportional fitting (dwelling x income)
│   ├── scripts/
│   │   ├── scrape_layer2_competitors.py - Overpass OSM competitor scraper
│   │   ├── build_layer2_corpus.py       - Qualitative knowledge corpus builder
│   │   └── ingest_layer2_chroma.py      - ChromaDB vector store ingestion
│   ├── data/
│   │   ├── sheets/                   - Official SingStat data (raw, cleaned, processed)
│   │   └── chroma_db/                - Persistent ChromaDB vector index
│   └── tests/                        - Automated pytest test suites (172 tests)
├── frontend/
│   ├── src/
│   │   ├── App.tsx                   - Main application workspace & simulation coordinator
│   │   ├── components/
│   │   │   ├── ai/                   - Persona message card, prompt input, reasoning tracker
│   │   │   ├── features/             - Top navigation, verdict banner, objection clusters
│   │   │   ├── figures/              - PSM curves (Recharts), cohort charts, sentiment gauge
│   │   │   └── ui/                   - Custom Toast dialogue, dialog, tabs, slider, buttons
│   │   ├── data/                     - Presets and sample output data
│   │   ├── lib/                      - Utility helpers (styling, cohort coloring, currency formatting)
│   │   ├── services/                 - API client, session management, rate limit handling
│   │   └── types/                    - TypeScript interfaces for API payloads
│   ├── package.json                  - Node.js dependencies and build scripts
│   └── vite.config.ts                - Vite configuration & dev API proxy
└── README.md                         - Documentation
```

---

## Offline Data Pipelines

To rebuild the sampling tables or refresh the ChromaDB vector database:

```bash
# Rebuild processed SingStat census tables
cd backend
python pipeline/clean_data.py
python pipeline/fit.py

# Re-index ChromaDB qualitative neighborhood knowledge
python scripts/scrape_layer2_competitors.py
python scripts/build_layer2_corpus.py
python scripts/ingest_layer2_chroma.py
```

---

## Automated Verification & Testing

### Backend Unit & Integration Tests (172 Tests)

```bash
cd backend
python -m pytest tests
```
The test suite validates:
- Rate limiter sliding-window accuracy, session isolation, and HTTP 429 headers.
- Test route authentication (`IS4401`) and rejection of unauthenticated requests.
- CORS configuration, HTTP security headers, and sanitized error responses.
- Monte Carlo persona sampling distribution matching SingStat census data.
- Van Westendorp PSM curve calculations and price monotonicity.
- End-to-end simulation pipelines with and without live API keys.

### Frontend Typechecking & Production Build

```bash
cd frontend
npm run build
```
Executes `tsc` typechecking followed by `vite build` minification.
