import type { EvidenceRequest, TranscriptAssessment } from "../types";

const REQUEST_TIMEOUT_MS = 10_000;

export async function analyzeEvidence(
  evidence: EvidenceRequest,
): Promise<TranscriptAssessment> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch("/api/v1/demo/analyze-evidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(evidence),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`The review service returned ${response.status}.`);
    }

    return (await response.json()) as TranscriptAssessment;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("The review took too long. Please try again.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}
