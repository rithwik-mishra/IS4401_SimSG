// Team package: preview works without a backend or OpenAI key.
// For real simulations, use mode: 'api' and your backend origin.
// Example local backend: http://127.0.0.1:8000 (on the viewer's computer).
// Never put an OpenAI API key here; configure it in backend/.env.
export const config = Object.freeze({
  mode: 'api',
  apiBaseUrl: 'http://127.0.0.1:8000',
  apiPath: '/api/simulate',
  timeoutMs: 180000,
});
