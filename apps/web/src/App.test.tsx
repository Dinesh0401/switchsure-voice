import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { TranscriptAssessment } from "./types";

const RESPONSE: TranscriptAssessment = {
  transcript_id: "test-1",
  text: "எனக்கு Metformin 500 mg வேணும்",
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
      surface: "வேணும்",
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

function mockSuccess(response: TranscriptAssessment = RESPONSE) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => response,
    }),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("SwitchSure review flow", () => {
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

  it("offers a retry when the service is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Service unavailable")));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Check Tamil–English sample" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Service unavailable");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
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
