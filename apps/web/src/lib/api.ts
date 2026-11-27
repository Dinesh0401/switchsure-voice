import type {
  ApiErrorResponse,
  EvidenceJobStatus,
  EvidenceRequest,
  TranscriptAssessment,
} from "../types";

const REQUEST_TIMEOUT_MS = 10_000;
const CAPABILITY_HEADER = "X-Job-Capability";

const SAFE_FALLBACK_MESSAGES: Record<string, string> = {
  invalid_request: "Please check the submitted information and try again.",
  invalid_evidence: "This evidence could not be checked safely.",
  job_not_found: "This temporary review is no longer available. Please start again.",
  invalid_job_state: "That action is no longer available for this review.",
  queue_unavailable: "The review service is temporarily unavailable.",
  queue_full: "The review service is busy. Please try again shortly.",
  result_unavailable: "This temporary result has expired. Please start again.",
  internal_error: "The review could not be completed safely.",
};

export class ApiClientError extends Error {
  readonly code: string;
  readonly status: number | null;
  readonly retryable: boolean;

  constructor(message: string, code: string, status: number | null, retryable: boolean) {
    super(message);
    this.name = "ApiClientError";
    this.code = code;
    this.status = status;
    this.retryable = retryable;
  }
}

function randomToken(byteCount: number): string {
  const bytes = new Uint8Array(byteCount);
  globalThis.crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Kept only in component memory. Callers must never persist or log this capability. */
export function createEphemeralJobCapability(): string {
  return randomToken(32);
}

function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<ApiErrorResponse>;
  return (
    typeof candidate.code === "string" &&
    typeof candidate.message === "string" &&
    typeof candidate.request_id === "string"
  );
}

async function requestJson<T>(
  path: string,
  init: RequestInit,
  externalSignal?: AbortSignal,
): Promise<T> {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  externalSignal?.addEventListener("abort", onAbort, { once: true });
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(path, { ...init, signal: controller.signal });
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok) {
      const code = isApiErrorResponse(body) ? body.code : "request_failed";
      const message =
        SAFE_FALLBACK_MESSAGES[code] ??
        "The review service could not complete this request.";
      throw new ApiClientError(
        message,
        code,
        response.status,
        response.status === 408 || response.status === 429 || response.status >= 500,
      );
    }
    return body as T;
  } catch (error) {
    if (error instanceof ApiClientError) throw error;
    if (controller.signal.aborted) {
      if (externalSignal?.aborted) {
        throw new ApiClientError("The request was cancelled.", "request_cancelled", null, true);
      }
      throw new ApiClientError("The review took too long. Please try again.", "timeout", null, true);
    }
    throw new ApiClientError(
      "The review service could not be reached. Check your connection and try again.",
      "network_error",
      null,
      true,
    );
  } finally {
    externalSignal?.removeEventListener("abort", onAbort);
    window.clearTimeout(timeout);
  }
}

export async function analyzeEvidence(
  evidence: EvidenceRequest,
): Promise<TranscriptAssessment> {
  return requestJson<TranscriptAssessment>("/api/v1/demo/analyze-evidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(evidence),
  });
}

function jobHeaders(capability: string): HeadersInit {
  return { "Content-Type": "application/json", [CAPABILITY_HEADER]: capability };
}

export function submitEvidenceJob(
  evidence: EvidenceRequest,
  capability: string,
  signal?: AbortSignal,
): Promise<EvidenceJobStatus> {
  return requestJson<EvidenceJobStatus>(
    "/api/v1/jobs/analyze-evidence",
    {
      method: "POST",
      headers: { ...jobHeaders(capability), "Idempotency-Key": randomToken(16) },
      body: JSON.stringify(evidence),
    },
    signal,
  );
}

export function getEvidenceJob(
  jobId: string,
  capability: string,
  signal?: AbortSignal,
): Promise<EvidenceJobStatus> {
  return requestJson<EvidenceJobStatus>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}`,
    { method: "GET", headers: { [CAPABILITY_HEADER]: capability } },
    signal,
  );
}

export function cancelEvidenceJob(
  jobId: string,
  capability: string,
  signal?: AbortSignal,
): Promise<EvidenceJobStatus> {
  return requestJson<EvidenceJobStatus>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST", headers: { [CAPABILITY_HEADER]: capability } },
    signal,
  );
}

export function retryEvidenceJob(
  jobId: string,
  evidence: EvidenceRequest,
  capability: string,
  signal?: AbortSignal,
): Promise<EvidenceJobStatus> {
  return requestJson<EvidenceJobStatus>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/retry`,
    {
      method: "POST",
      headers: jobHeaders(capability),
      body: JSON.stringify(evidence),
    },
    signal,
  );
}
