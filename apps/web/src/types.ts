export type DecisionAction = "accept" | "review" | "defer";

export interface LanguageProbability {
  code: string;
  probability: number;
}

export interface WordAssessment {
  index: number;
  surface: string;
  start_ms: number;
  end_ms: number;
  recognition_risk: number;
  boundary_uncertainty: number;
  languages: LanguageProbability[];
}

export interface EntityAssessment {
  start_word: number;
  end_word: number;
  entity_type: string;
  surface: string;
  canonical_value: string | null;
  criticality: 1 | 2 | 4;
  preservation_risk: number;
}

export interface TranscriptAssessment {
  transcript_id: string;
  text: string;
  action: DecisionAction;
  reason_codes: string[];
  calibration_status: "uncalibrated_baseline" | "calibrated";
  model_version: string;
  risk_model_version: string;
  policy_version: string;
  words: WordAssessment[];
  entities: EntityAssessment[];
}

export interface EvidenceRequest {
  model_version: string;
  words: Array<{
    surface: string;
    start_ms: number;
    end_ms: number;
    decoder_confidence: number;
    acoustic_quality?: number;
    boundary_probability?: number;
    languages: LanguageProbability[];
  }>;
}
