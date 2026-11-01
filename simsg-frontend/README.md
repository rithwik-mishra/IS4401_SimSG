⏱ About 13 minutes to read

# SimSG frontend — team handoff

This package contains the desktop-first HTML/CSS/JavaScript frontend. It includes a saved response for UI review and a connector for the existing SimSG backend. No frontend build or package installation is required.

## 1. Review the page locally

Install Node.js 20 or later if it is not already available. Extract the ZIP, open a terminal in its `simsg-frontend` folder, and run:

```bash
npm start
```

Wait for `Local: http://127.0.0.1:4173`, then open that address in your browser. Keep the terminal running; Ctrl+C stops the server. Do not double-click `dist/index.html`: file:// access can prevent JavaScript modules and fetched data from loading.

If port 4173 is already in use, open the existing server or run:

```bash
node scripts/serve.mjs --port 4174
```

Then open http://127.0.0.1:4174 instead. Run `npm run check` for syntax and request-contract checks.

The distributed ZIP defaults to **demo mode**. It displays the saved 30-person response; pressing Simulate reloads that response. Editing the idea, price, location, or respondent count does not generate new results in this mode. The page itself intentionally omits demo annotations. No backend or OpenAI key is needed for this UI review.

## 2. Connect a real backend

The backend is a separate project and is not included in this ZIP. Its maintainer must install its Python dependencies, supply its data/vector index, configure credentials, and start it. Check the backend's `/docs` page before testing the frontend. The existing repository must be launched from its root using the module `backend.app.main:app`, because some routes import the `backend` package.

**OpenAI API keys belong in the backend's `.env` as `OPENAI_API_KEY=...`. Never put a provider key in config.js or any other frontend file.** apiBaseUrl is the SimSG server address, not the OpenAI API address.

API mode sends POST /api/simulate only when Simulate is clicked. It sends JSON with raw_query, sample_size, use_mock, and any supplied override_price, override_town, or seed. The response must contain recommendations, personas, and statistics.

For an initial connection check, turn off Use AI responses and submit. After backend credentials are configured, enable AI responses for a model-backed run. The existing backend can fall back to rule-based responses when a model call fails; receiving a report alone does not prove OpenAI was used. Model-backed requests may incur usage charges.

## 3. Upload or distribute

- For source review or development, send the whole `simsg-frontend-source.zip`.
- For static hosting, upload the **contents of dist/**, so index.html is at the site root. No build command is needed. Preserve assets, js, config.js, styles.css, and sample-output.json.
- Leave demo mode enabled for a UI-only hosted preview. To offer real simulations, configure a reachable HTTPS backend and allow the frontend origin in the backend's CORS settings.
- A same-origin deployment may leave apiBaseUrl empty only when a reverse proxy routes /api to the Python backend. A static host alone does not run that backend.

The ZIP excludes Git metadata, owner-specific hosting settings, Python environments, backend data, and `.env` files. Do not add those when forwarding it. Each teammate supplies their own environment or uses the team's shared backend. The original private hosted preview does not grant access merely by forwarding its URL.

## Files and behavior

`dist/index.html` and `styles.css` define the layout. `dist/js/api.js` handles requests; app.js renders reports; personas.js handles card details; charts.js renders charts; format.js contains display helpers. `scripts/serve.mjs` serves the local preview.

More options expands the input fields. Persona cards show details on hover, focus, or click. Composition and purchase intent rank groups descending. Response totals describe the completed report, not later edits to the input. Price curves use the returned data; the frontend does not rerun statistical calculations.

Browser interactions and request handling have been tested with intercepted responses. The separate local backend has passed startup and API-schema checks; a paid end-to-end OpenAI simulation has not been verified as part of this handoff.
