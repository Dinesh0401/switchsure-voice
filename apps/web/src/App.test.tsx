import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { EvidenceJobStatus, JobState, TranscriptAssessment } from "./types";

const RESPONSE: TranscriptAssessment = {
  transcript_id: "test-1",
  text: "எனக்கு Metformin 500 mg வேண்டும்",
  action: "review",
  reason_codes: ["critical-entity:dosage"],
  calibration_status: "uncalibrated_baseline",
  model_version: "fixture-v1",
  risk_model_version: "heuristic-v1",
  policy_version: "cost-policy-v1",
  words: [
    {
      index: 0,
      surface: "எனக்கு",
      start_ms: 0,
      end_ms: 300,
      recognition_risk: 0.08,
      boundary_uncertainty: 0.2,
      languages: [{ code: "ta", probability: 1 }],
    },
    {
      index: 1,
      surface: "Metformin",
      start_ms: 310,
      end_ms: 700,
      recognition_risk: 0.28,
      boundary_uncertainty: 0.76,
      languages: [{ code: "en", probability: 1 }],
    },
    {
      index: 2,
      surface: "500",
      start_ms: 710,
      end_ms: 900,
      recognition_risk: 0.71,
      boundary_uncertainty: 0.99,
      languages: [{ code: "neutral", probability: 1 }],
    },
    {
      index: 3,
      surface: "mg",
      start_ms: 910,
      end_ms: 1030,
      recognition_risk: 0.62,
      boundary_uncertainty: 0.98,
      languages: [{ code: "en", probability: 1 }],
    },
    {
      index: 4,
      surface: "வேண்டும்",
      start_ms: 1040,
      end_ms: 1400,
      recognition_risk: 0.1,
      boundary_uncertainty: 0.3,
      languages: [{ code: "ta", probability: 1 }],
    },
  ],
  entities: [
    {
      start_word: 2,
      end_word: 3,
      entity_type: "dosage",
      surface: "500 mg",
      canonical_value: "500 mg",
      criticality: 4,
      preservation_risk: 0.74,
    },
  ],
};

function job(
  state: JobState,
  result: TranscriptAssessment | null = null,
  overrides: Partial<EvidenceJobStatus> = {},
): EvidenceJobStatus {
  return {
    job_id: "job-1",
    state,
    created_at: "2026-11-25T00:00:00Z",
    updated_at: "2026-11-25T00:00:01Z",
    expires_at: "2026-11-25T00:15:00Z",
    attempts: 1,
    max_attempts: 3,
    retention_class: "ephemeral",
    error_code: null,
    result,
    ...overrides,
  };
}

function response(body: unknown, ok = true, status = 200) {
  return { ok, status, json: async () => body };
}

function mockSuccess(result: TranscriptAssessment = RESPONSE) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response(job("succeeded", result), true, 202)));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("SwitchSure asynchronous review flow", () => {
  it("focuses the user on the critical entity and supports undo", async () => {
    mockSuccess();
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));

    expect(
      await screen.findByRole("heading", { name: "Please check 1 critical detail" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Check dosage")).toHaveValue("500 mg");
    expect(screen.getByRole("button", { name: "Mark transcript reviewed" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Confirm dosage" }));
    expect(screen.getByText("Confirmed")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mark transcript reviewed" })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Undo confirmation" }));
    expect(screen.getByText("Needs your check")).toBeInTheDocument();
  });

  it("polls queued and processing jobs until a result is ready", async () => {
    vi.useFakeTimers();
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response(job("queued"), true, 202))
      .mockResolvedValueOnce(response(job("running")))
      .mockResolvedValueOnce(response(job("succeeded", RESPONSE)));
    vi.stubGlobal("fetch", fetchMock);
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));
    await act(async () => undefined);
    expect(screen.getByRole("heading", { name: "Waiting for an available checker" })).toBeVisible();

    await act(async () => {
      vi.advanceTimersByTime(800);
      await Promise.resolve();
    });
    expect(screen.getByRole("heading", { name: "Checking the sample" })).toBeVisible();

    await act(async () => {
      vi.advanceTimersByTime(800);
      await Promise.resolve();
    });
    expect(screen.getByRole("heading", { name: "Please check 1 critical detail" })).toBeVisible();
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("cancels an active temporary job", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response(job("queued"), true, 202))
      .mockResolvedValueOnce(response(job("cancelled")));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));
    await user.click(await screen.findByRole("button", { name: "Cancel check" }));

    expect(screen.getByRole("heading", { name: "The sample check was cancelled" })).toBeVisible();
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/v1/jobs/job-1/cancel");
  });

  it("retries a failed job with fresh evidence", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        response(job("failed", null, { error_code: "model_timeout", attempts: 1 }), true, 202),
      )
      .mockResolvedValueOnce(response(job("succeeded", RESPONSE), true, 202));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("took too long");
    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(
      await screen.findByRole("heading", { name: "Please check 1 critical detail" }),
    ).toBeVisible();
    expect(fetchMock.mock.calls[1]?.[0]).toBe("/api/v1/jobs/job-1/retry");
  });

  it("keeps the capability ephemeral and offers a typed fallback", async () => {
    const storageSpy = vi.spyOn(Storage.prototype, "setItem");
    const fetchMock = vi.fn().mockRejectedValue(new Error("offline"));
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("Typed transcript"), "நாளை 10 AM appointment");
    await user.click(screen.getByRole("button", { name: "Use typed transcript" }));
    expect(screen.getByRole("button", { name: "Typed transcript ready" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("could not be reached");
    const headers = (fetchMock.mock.calls[0]?.[1] as RequestInit).headers as Record<string, string>;
    expect(headers["X-Job-Capability"]).toHaveLength(64);
    expect(storageSpy).not.toHaveBeenCalled();
  });

  it("does not display unrecognized server error details", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        response(
          { code: "unrecognized_failure", message: "private internal model detail", request_id: "r" },
          false,
          500,
        ),
      ),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Check Tamil/ }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("could not complete this request");
    expect(alert).not.toHaveTextContent("private internal model detail");
  });

  it("describes transcript uncertainty accurately when no entity is detected", async () => {
    mockSuccess({
      ...RESPONSE,
      action: "review",
      reason_codes: ["high-word-risk"],
      entities: [],
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));

    expect(
      await screen.findByRole("heading", { name: "Please check the uncertain transcript" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("0 critical details")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Mark transcript reviewed" })).toBeEnabled();
  });

  it("has no automated accessibility violations in the review state", async () => {
    mockSuccess();
    const user = userEvent.setup();
    const { container } = render(<App />);

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));
    await screen.findByRole("heading", { name: "Please check 1 critical detail" });

    const results = await axe(container);
    expect(results.violations).toEqual([]);
  });
});
