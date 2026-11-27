import { useCallback, useEffect, useRef, useState } from "react";

import type { EvidenceJobStatus, EvidenceRequest, TranscriptAssessment } from "../types";
import {
  ApiClientError,
  cancelEvidenceJob,
  createEphemeralJobCapability,
  getEvidenceJob,
  retryEvidenceJob,
  submitEvidenceJob,
} from "./api";

const POLL_INTERVAL_MS = 750;

export type AsyncReviewState =
  | "idle"
  | "submitting"
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

type FailureKind = "request" | "job" | null;

export interface AsyncReview {
  state: AsyncReviewState;
  result: TranscriptAssessment | null;
  error: string;
  retryable: boolean;
  canCancel: boolean;
  submit: () => Promise<void>;
  cancel: () => Promise<void>;
  retry: () => Promise<void>;
}

function safeJobFailureMessage(errorCode: string | null): string {
  const messages: Record<string, string> = {
    model_timeout: "The speech check took too long.",
    worker_lease_timeout: "The speech check stopped before it could finish.",
    process_restart_ephemeral_data_unavailable:
      "The temporary evidence was cleared after a service restart.",
  };
  return messages[errorCode ?? ""] ?? "The speech check could not be completed safely.";
}

function errorDetails(cause: unknown): { message: string; retryable: boolean } {
  if (cause instanceof ApiClientError) {
    return { message: cause.message, retryable: cause.retryable };
  }
  return { message: "The speech check could not be completed safely.", retryable: true };
}

export function useEvidenceJob(evidence: EvidenceRequest): AsyncReview {
  const [state, setState] = useState<AsyncReviewState>("idle");
  const [result, setResult] = useState<TranscriptAssessment | null>(null);
  const [error, setError] = useState("");
  const [retryable, setRetryable] = useState(false);
  const capabilityRef = useRef<string | null>(null);
  const jobRef = useRef<EvidenceJobStatus | null>(null);
  const failureKindRef = useRef<FailureKind>(null);

  const applyStatus = useCallback((job: EvidenceJobStatus) => {
    jobRef.current = job;
    setError("");
    setRetryable(false);
    failureKindRef.current = null;

    if (job.state === "succeeded") {
      if (!job.result) {
        setState("failed");
        setError("The temporary result is unavailable. Please start again.");
        setRetryable(true);
        failureKindRef.current = "request";
        return;
      }
      setResult(job.result);
      setState("succeeded");
      return;
    }

    setResult(null);
    setState(job.state);
    if (job.state === "failed") {
      setError(safeJobFailureMessage(job.error_code));
      setRetryable(job.attempts < job.max_attempts);
      failureKindRef.current = "job";
    }
  }, []);

  const failRequest = useCallback((cause: unknown) => {
    const details = errorDetails(cause);
    setState("failed");
    setResult(null);
    setError(details.message);
    setRetryable(details.retryable);
    failureKindRef.current = "request";
  }, []);

  const submit = useCallback(async () => {
    const capability = createEphemeralJobCapability();
    capabilityRef.current = capability;
    jobRef.current = null;
    failureKindRef.current = null;
    setState("submitting");
    setResult(null);
    setError("");
    setRetryable(false);
    try {
      applyStatus(await submitEvidenceJob(evidence, capability));
    } catch (cause) {
      failRequest(cause);
    }
  }, [applyStatus, evidence, failRequest]);

  useEffect(() => {
    if (state !== "queued" && state !== "running") return;
    const job = jobRef.current;
    const capability = capabilityRef.current;
    if (!job || !capability) return;

    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      void getEvidenceJob(job.job_id, capability, controller.signal)
        .then(applyStatus)
        .catch((cause: unknown) => {
          if (!controller.signal.aborted) failRequest(cause);
        });
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [applyStatus, failRequest, state]);

  const cancel = useCallback(async () => {
    const job = jobRef.current;
    const capability = capabilityRef.current;
    if (!job || !capability || (state !== "queued" && state !== "running")) return;
    try {
      applyStatus(await cancelEvidenceJob(job.job_id, capability));
    } catch (cause) {
      failRequest(cause);
    }
  }, [applyStatus, failRequest, state]);

  const retry = useCallback(async () => {
    if (!retryable) return;
    const job = jobRef.current;
    const capability = capabilityRef.current;
    setState("submitting");
    setError("");
    setRetryable(false);
    try {
      if (failureKindRef.current === "job" && job && capability) {
        applyStatus(await retryEvidenceJob(job.job_id, evidence, capability));
      } else if (job && capability) {
        applyStatus(await getEvidenceJob(job.job_id, capability));
      } else {
        await submit();
      }
    } catch (cause) {
      failRequest(cause);
    }
  }, [applyStatus, evidence, failRequest, retryable, submit]);

  return {
    state,
    result,
    error,
    retryable,
    canCancel: state === "queued" || state === "running",
    submit,
    cancel,
    retry,
  };
}
