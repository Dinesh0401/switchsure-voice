import type { EvidenceRequest } from "../types";

export const SAMPLE_EVIDENCE: EvidenceRequest = {
  model_version: "tamil-english-demo-v1",
  words: [
    {
      surface: "எனக்கு",
      start_ms: 0,
      end_ms: 340,
      decoder_confidence: 0.95,
      acoustic_quality: 0.94,
      boundary_probability: 0.08,
      languages: [
        { code: "ta", probability: 0.96 },
        { code: "en", probability: 0.04 },
      ],
    },
    {
      surface: "Metformin",
      start_ms: 350,
      end_ms: 790,
      decoder_confidence: 0.76,
      acoustic_quality: 0.82,
      boundary_probability: 0.78,
      languages: [
        { code: "en", probability: 0.71 },
        { code: "neutral", probability: 0.29 },
      ],
    },
    {
      surface: "500",
      start_ms: 800,
      end_ms: 1030,
      decoder_confidence: 0.23,
      acoustic_quality: 0.58,
      boundary_probability: 0.53,
      languages: [
        { code: "neutral", probability: 0.55 },
        { code: "en", probability: 0.45 },
      ],
    },
    {
      surface: "mg",
      start_ms: 1040,
      end_ms: 1210,
      decoder_confidence: 0.38,
      acoustic_quality: 0.64,
      boundary_probability: 0.49,
      languages: [
        { code: "en", probability: 0.86 },
        { code: "ta", probability: 0.14 },
      ],
    },
    {
      surface: "வேணும்",
      start_ms: 1220,
      end_ms: 1620,
      decoder_confidence: 0.91,
      acoustic_quality: 0.9,
      boundary_probability: 0.81,
      languages: [
        { code: "ta", probability: 0.93 },
        { code: "en", probability: 0.07 },
      ],
    },
  ],
};
