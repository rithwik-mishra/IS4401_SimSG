import { SimulateRequest, SimulateResponse } from '../types/api';
import { SAMPLE_SIMULATION_RESULT } from '../data/sampleOutput';

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  (typeof window !== 'undefined' && window.location.hostname.includes('simsg.vercel.app')
    ? 'https://simsg-backend.vercel.app'
    : '');

export async function runSimulation(
  request: SimulateRequest,
  options?: { signal?: AbortSignal; allowMockFallback?: boolean }
): Promise<SimulateResponse> {
  // If mock mode is explicitly requested, simulate a realistic processing delay
  if (request.use_mock) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    return SAMPLE_SIMULATION_RESULT;
  }

  const endpoint = `${API_BASE_URL}/api/simulate`;

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
      signal: options?.signal,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let errorDetail = `Simulation request failed with status ${response.status}`;
      try {
        const errorJson = JSON.parse(errorText);
        if (errorJson.detail) {
          errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
        }
      } catch {
        if (errorText) errorDetail += `: ${errorText.slice(0, 150)}`;
      }
      throw new Error(errorDetail);
    }

    const data: SimulateResponse = await response.json();

    if (!data.recommendations || !Array.isArray(data.personas) || !data.statistics) {
      throw new Error('Received malformed response from simulation engine: missing core sections.');
    }

    return data;
  } catch (err: any) {
    if (err.name === 'AbortError') {
      throw err;
    }
    // If fallback is permitted and backend is offline during preview, fallback gracefully with notice
    if (options?.allowMockFallback) {
      console.warn('Backend unavailable, falling back to local simulation sample:', err.message);
      return SAMPLE_SIMULATION_RESULT;
    }
    throw err;
  }
}

