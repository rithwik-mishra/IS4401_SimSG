export class SimulationError extends Error {}

export function buildRequest(form) {
  const values = new FormData(form);
  const raw_query = String(values.get('raw_query') || '').trim();
  if (!raw_query) throw new SimulationError('Describe your business idea first.');
  const sample_size = Number(values.get('sample_size'));
  if (!Number.isInteger(sample_size) || sample_size < 5 || sample_size > 100) throw new SimulationError('Choose between 5 and 100 respondents.');
  const request = {raw_query, sample_size, use_mock: !values.has('use_ai')};
  const price = String(values.get('override_price') ?? '').trim();
  if (price !== '') {
    const numeric = Number(price);
    if (!Number.isFinite(numeric) || numeric < 0) throw new SimulationError('Enter a valid price of S$0 or more.');
    request.override_price = numeric;
  }
  const town = String(values.get('override_town') || '');
  if (town) request.override_town = town;
  const seed = String(values.get('seed') ?? '').trim();
  if (seed !== '') {
    const numeric = Number(seed);
    if (!Number.isSafeInteger(numeric)) throw new SimulationError('Enter a whole-number seed within the supported range.');
    request.seed = numeric;
  }
  return request;
}

export function validateResponse(data) {
  if (!data || typeof data !== 'object' || !data.recommendations || !Array.isArray(data.personas) || !data.statistics || typeof data.statistics !== 'object') {
    throw new SimulationError('The response is missing the expected recommendations, personas, or statistics.');
  }
  return data;
}

function errorDetail(data, status) {
  if (typeof data?.detail === 'string') return data.detail;
  if (Array.isArray(data?.detail)) return data.detail.map(item => item.msg || 'Invalid input').join('; ');
  return `The simulation could not be completed (HTTP ${status}). Please try again.`;
}

export async function runSimulation(request, config, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), config.timeoutMs);
  try {
    let response;
    if (config.mode === 'demo') {
      response = await fetch(new URL('../sample-output.json', import.meta.url), {signal: controller.signal});
    } else if (config.mode === 'api') {
      const base = config.apiBaseUrl.replace(/\/$/, '');
      response = await fetch(base + config.apiPath, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(request), signal: controller.signal,
      });
    } else throw new SimulationError('This preview has an unsupported connection setting.');
    let data;
    try { data = await response.json(); }
    catch { throw new SimulationError('The server did not return a readable JSON response.'); }
    if (!response.ok) throw new SimulationError(errorDetail(data, response.status));
    return validateResponse(data);
  } catch (error) {
    if (error.name === 'AbortError') throw new SimulationError('The request took too long. The server may still be processing it; retry only when ready.');
    if (error instanceof SimulationError) throw error;
    throw new SimulationError(config.mode === 'demo' ? 'The saved example could not be loaded. Please refresh or try again.' : 'Could not reach the simulation service. Check that the backend is running and allows requests from this page.');
  } finally { clearTimeout(timer); }
}
