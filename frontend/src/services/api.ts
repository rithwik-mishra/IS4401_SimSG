import { SimulateRequest, SimulateResponse } from '../types/api';
import { SAMPLE_SIMULATION_RESULT } from '../data/sampleOutput';

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  (typeof window !== 'undefined' && window.location.hostname.includes('simsg.vercel.app')
    ? 'https://simsg-backend.vercel.app'
    : '');

/**
 * Retrieves or initializes a persistent anonymous session ID stored in localStorage.
 * Used for hourly rate-limiting of simulations per unique browser session.
 */
export function getOrCreateSessionId(): string {
  if (typeof window === 'undefined') return 'server-session';
  const STORAGE_KEY = 'simsg_session_id';
  try {
    let sessionId = localStorage.getItem(STORAGE_KEY);
    if (!sessionId || sessionId.trim().length === 0) {
      sessionId =
        typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
          ? crypto.randomUUID()
          : `sess_${Date.now()}_${Math.random().toString(36).substring(2, 10)}`;
      localStorage.setItem(STORAGE_KEY, sessionId);
    }
    return sessionId;
  } catch {
    return 'ephemeral-session';
  }
}

export interface AIStatus {
  persona_engine: 'llm' | 'rule_based';
  llm_provider: string | null;
  llm_model: string | null;
  reason: string | null;
}

export type ApiRejectionType =
  | 'pii'
  | 'malicious'
  | 'irrelevant'
  | 'validation'
  | 'rate_limit'
  | 'general';

export class ApiError extends Error {
  status: number;
  isRateLimit: boolean;
  isValidationError: boolean;
  retryAfter?: number;
  rejectionType: ApiRejectionType;

  constructor(
    message: string,
    status: number,
    options?: {
      isRateLimit?: boolean;
      isValidationError?: boolean;
      retryAfter?: number;
      rejectionType?: ApiRejectionType;
    }
  ) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.isRateLimit = options?.isRateLimit ?? (status === 429);
    this.isValidationError = options?.isValidationError ?? (status === 422);
    this.retryAfter = options?.retryAfter;
    this.rejectionType = options?.rejectionType ?? (status === 429 ? 'rate_limit' : status === 422 ? 'validation' : 'general');
  }
}

/** Which AI model the backend would use right now (or none: backup logic). */
export async function getAIStatus(signal?: AbortSignal): Promise<AIStatus> {
  const response = await fetch(`${API_BASE_URL}/api/ai-status`, { signal });
  if (!response.ok) throw new Error(`AI status request failed with status ${response.status}`);
  return response.json();
}

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
    const sessionId = getOrCreateSessionId();
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': sessionId,
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
          if (typeof errorJson.detail === 'string') {
            errorDetail = errorJson.detail;
          } else if (Array.isArray(errorJson.detail)) {
            errorDetail = errorJson.detail
              .map((item: any) =>
                typeof item === 'string'
                  ? item
                  : item?.msg
                  ? `${item.msg}${item?.loc ? ` (${item.loc.join('.')})` : ''}`
                  : JSON.stringify(item)
              )
              .join('; ');
          } else {
            errorDetail = JSON.stringify(errorJson.detail);
          }
        }
      } catch {
        if (errorText) errorDetail += `: ${errorText.slice(0, 150)}`;
      }

      const isRateLimit = response.status === 429;
      const isValidationError = response.status === 422;
      let retryAfter: number | undefined;

      if (isRateLimit) {
        const retryHeader = response.headers.get('Retry-After');
        if (retryHeader) {
          retryAfter = parseInt(retryHeader, 10);
        }
      }

      let rejectionType: ApiRejectionType = isRateLimit
        ? 'rate_limit'
        : isValidationError
        ? 'validation'
        : 'general';

      const lower = errorDetail.toLowerCase();
      if (
        lower.includes('personally identifiable') ||
        lower.includes('pii') ||
        lower.includes('nric') ||
        lower.includes('credit card') ||
        lower.includes('email address') ||
        lower.includes('phone number') ||
        lower.includes('passport')
      ) {
        rejectionType = 'pii';
      } else if (
        lower.includes('unsafe') ||
        lower.includes('malicious') ||
        lower.includes('injection') ||
        lower.includes('script')
      ) {
        rejectionType = 'malicious';
      } else if (
        lower.includes("doesn't appear to describe") ||
        lower.includes('business concept') ||
        lower.includes('failed to decompose') ||
        lower.includes('clearer business pitch')
      ) {
        rejectionType = 'irrelevant';
      }

      const error = new ApiError(errorDetail, response.status, {
        isRateLimit,
        isValidationError,
        retryAfter,
        rejectionType,
      });

      // Backward-compatibility properties
      (error as any).isRateLimit = isRateLimit;
      (error as any).status = response.status;
      (error as any).retryAfter = retryAfter;

      throw error;
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
    // Never fall back to mock data if rejected by rate limiter (429), validation guardrails (422), or any 4xx client status
    if (
      err.isRateLimit ||
      err.status === 429 ||
      err.isValidationError ||
      err.status === 422 ||
      (typeof err.status === 'number' && err.status >= 400 && err.status < 500)
    ) {
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

