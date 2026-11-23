import { useMemo, useState } from "react";

import { analyzeEvidence } from "./lib/api";
import { SAMPLE_EVIDENCE } from "./lib/sample";
import type { EntityAssessment, TranscriptAssessment, WordAssessment } from "./types";

type RequestState = "idle" | "loading" | "success" | "error";

const ENTITY_LABELS: Record<string, string> = {
  dosage: "dosage",
  medication: "medicine name",
  phone: "phone number",
  email: "email address",
  identifier: "reference number",
  date: "date",
  time: "time",
  money: "amount",
};

function languageName(code: string): string {
  const names: Record<string, string> = {
    ta: "Tamil",
    en: "English",
    neutral: "Language-neutral",
  };
  return names[code] ?? code.toUpperCase();
}

function dominantLanguage(word: WordAssessment): string {
  const best = [...word.languages].sort((left, right) => right.probability - left.probability)[0];
  return best ? languageName(best.code) : "Language not identified";
}

function wordState(word: WordAssessment): "clear" | "check" | "uncertain" {
  if (word.recognition_risk >= 0.55) return "uncertain";
  if (word.recognition_risk >= 0.25 || word.boundary_uncertainty >= 0.8) return "check";
  return "clear";
}

function actionCopy(result: TranscriptAssessment): { eyebrow: string; heading: string; body: string } {
  const count = result.entities.length;
  if (result.action === "defer") {
    return {
      eyebrow: "More input needed",
      heading:
        count > 0
          ? "Please repeat or type the critical detail"
          : "Please repeat or type the uncertain words",
      body:
        count > 0
          ? "The available evidence is not strong enough to safely accept this value."
          : "The available evidence is not strong enough to safely accept this transcript.",
    };
  }
  if (result.action === "review") {
    return {
      eyebrow: "Focused review",
      heading:
        count > 0
          ? `Please check ${count} critical ${count === 1 ? "detail" : "details"}`
          : "Please check the uncertain transcript",
      body:
        count > 0
          ? "The rest of the transcript stays visible while you confirm only the part that matters."
          : "No critical entity was detected, but the overall evidence still needs your review.",
    };
  }
  return {
    eyebrow: "No critical review triggered",
    heading: "The transcript is ready for your review",
    body: "Read the text once before using it. SwitchSure never submits it automatically.",
  };
}

export function App() {
  const [requestState, setRequestState] = useState<RequestState>("idle");
  const [result, setResult] = useState<TranscriptAssessment | null>(null);
  const [error, setError] = useState("");
  const [entityDrafts, setEntityDrafts] = useState<Record<number, string>>({});
  const [confirmedEntities, setConfirmedEntities] = useState<Set<number>>(new Set());
  const [transcriptDraft, setTranscriptDraft] = useState("");
  const [finalized, setFinalized] = useState(false);

  const action = useMemo(() => (result ? actionCopy(result) : null), [result]);

  async function runReview() {
    setRequestState("loading");
    setError("");
    setFinalized(false);
    try {
      const next = await analyzeEvidence(SAMPLE_EVIDENCE);
      setResult(next);
      setTranscriptDraft(next.text);
      setEntityDrafts(
        Object.fromEntries(next.entities.map((entity, index) => [index, entity.surface])),
      );
      setConfirmedEntities(new Set());
      setRequestState("success");
    } catch (cause) {
      setRequestState("error");
      setError(cause instanceof Error ? cause.message : "The review could not be completed.");
    }
  }

  function updateEntity(index: number, value: string) {
    setEntityDrafts((current) => ({ ...current, [index]: value }));
    setConfirmedEntities((current) => {
      const next = new Set(current);
      next.delete(index);
      return next;
    });
    setFinalized(false);
  }

  function confirmEntity(index: number) {
    setConfirmedEntities((current) => new Set(current).add(index));
    setFinalized(false);
  }

  function undoEntity(index: number) {
    setConfirmedEntities((current) => {
      const next = new Set(current);
      next.delete(index);
      return next;
    });
    setFinalized(false);
  }

  const allCriticalConfirmed =
    !result || result.entities.length === 0 || confirmedEntities.size === result.entities.length;

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="SwitchSure Voice home">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span>
            <strong>SwitchSure</strong>
            <small>Voice</small>
          </span>
        </a>
        <span className="build-badge">Research build 0.1</span>
      </header>

      <main id="main-content">
        <section className="hero" id="top" aria-labelledby="hero-title">
          <div className="hero-copy">
            <p className="eyebrow">Tamil + English · risk-aware speech</p>
            <h1 id="hero-title">Hear every switch. Protect every critical detail.</h1>
            <p className="hero-lede">
              SwitchSure turns multilingual speech into a transcript that knows when to pause and
              ask—not one that hides uncertainty behind polished text.
            </p>
            <div className="hero-actions">
              <button
                className="primary-button"
                type="button"
                onClick={runReview}
                disabled={requestState === "loading"}
              >
                {requestState === "loading" ? "Checking sample…" : "Check Tamil–English sample"}
              </button>
              <a className="text-link" href="#how-it-works">
                See what is checked <span aria-hidden="true">↓</span>
              </a>
            </div>
            <p className="prototype-note">
              This vertical slice reviews model evidence. Live microphone transcription connects
              after the ASR baseline passes the evaluation gates.
            </p>
          </div>

          <aside className="assurance-card" aria-labelledby="assurance-title">
            <p className="card-kicker">Safety contract</p>
            <h2 id="assurance-title">A transcript is not a plain string.</h2>
            <ul className="assurance-list">
              <li>
                <span className="assurance-number">01</span>
                <span>Switch boundaries stay visible as evidence.</span>
              </li>
              <li>
                <span className="assurance-number">02</span>
                <span>Dosages, dates and identifiers get higher protection.</span>
              </li>
              <li>
                <span className="assurance-number">03</span>
                <span>Low evidence leads to review or repeat—not a guess.</span>
              </li>
            </ul>
          </aside>
        </section>

        <div className="live-region" aria-live="polite" aria-atomic="true">
          {requestState === "loading" && "Checking the sample transcript."}
          {requestState === "success" && action?.heading}
          {requestState === "error" && error}
          {finalized && "Transcript marked as reviewed."}
        </div>

        {requestState === "error" && (
          <section className="error-panel" role="alert" aria-labelledby="error-heading">
            <div>
              <p className="card-kicker">Connection problem</p>
              <h2 id="error-heading">We could not check the sample</h2>
              <p>{error}</p>
            </div>
            <button className="secondary-button" type="button" onClick={runReview}>
              Try again
            </button>
          </section>
        )}

        {result && requestState === "success" && action && (
          <section className="review-workspace" aria-labelledby="review-title">
            <div className={`decision-banner decision-${result.action}`}>
              <div className="decision-icon" aria-hidden="true">
                {result.action === "accept" ? "✓" : result.action === "review" ? "!" : "↻"}
              </div>
              <div>
                <p className="card-kicker">{action.eyebrow}</p>
                <h2 id="review-title">{action.heading}</h2>
                <p>{action.body}</p>
              </div>
            </div>

            {result.calibration_status === "uncalibrated_baseline" && (
              <p className="research-warning">
                <strong>Research preview:</strong> the current risk scorer is transparent but not
                yet calibrated on speaker-disjoint Tamil–English data. Do not use this build for a
                real medical or financial decision.
              </p>
            )}

            <div className="workspace-grid">
              <article className="transcript-panel" aria-labelledby="transcript-title">
                <div className="panel-heading">
                  <div>
                    <p className="card-kicker">Transcript evidence</p>
                    <h3 id="transcript-title">What SwitchSure heard</h3>
                  </div>
                  <div className="legend" aria-label="Transcript key">
                    <span><i className="legend-clear" />Clear</span>
                    <span><i className="legend-check" />Check</span>
                    <span><i className="legend-uncertain" />Uncertain</span>
                  </div>
                </div>

                <p className="screen-reader-only">{result.text}</p>
                <div className="word-line" aria-hidden="true">
                  {result.words.map((word) => (
                    <span className={`word-token word-${wordState(word)}`} key={word.index}>
                      <span>{word.surface}</span>
                      <small>{dominantLanguage(word)}</small>
                    </span>
                  ))}
                </div>

                <div className="typed-fallback">
                  <label htmlFor="transcript-edit">Edit the full transcript</label>
                  <p id="transcript-edit-help">
                    Typing is always available as an equal alternative to speech.
                  </p>
                  <textarea
                    id="transcript-edit"
                    aria-describedby="transcript-edit-help"
                    rows={4}
                    value={transcriptDraft}
                    onChange={(event) => {
                      setTranscriptDraft(event.target.value);
                      setFinalized(false);
                    }}
                  />
                </div>
              </article>

              <aside className="review-queue" aria-labelledby="queue-title">
                <div className="panel-heading compact-heading">
                  <div>
                    <p className="card-kicker">Critical detail queue</p>
                    <h3 id="queue-title">Confirm before use</h3>
                  </div>
                  <span className="count-badge">{result.entities.length}</span>
                </div>

                {result.entities.length === 0 ? (
                  <p className="empty-state">No structured critical value was detected.</p>
                ) : (
                  <ol className="entity-list">
                    {result.entities.map((entity, index) => (
                      <EntityReview
                        key={`${entity.entity_type}-${entity.start_word}`}
                        entity={entity}
                        index={index}
                        value={entityDrafts[index] ?? entity.surface}
                        confirmed={confirmedEntities.has(index)}
                        onChange={updateEntity}
                        onConfirm={confirmEntity}
                        onUndo={undoEntity}
                      />
                    ))}
                  </ol>
                )}
              </aside>
            </div>

            <div className="review-footer">
              <p>
                {allCriticalConfirmed
                  ? "Critical details checked. Read the full transcript once before marking it reviewed."
                  : "Confirm each critical detail, or type a correction before continuing."}
              </p>
              <button
                className="primary-button"
                type="button"
                disabled={!allCriticalConfirmed || finalized}
                onClick={() => setFinalized(true)}
              >
                {finalized ? "Reviewed" : "Mark transcript reviewed"}
              </button>
            </div>
          </section>
        )}

        <section className="how-it-works" id="how-it-works" aria-labelledby="how-title">
          <p className="eyebrow">A real-world decision pipeline</p>
          <h2 id="how-title">Fast where it can be. Careful where it must be.</h2>
          <div className="method-grid">
            <article>
              <span>01</span>
              <h3>Listen & align</h3>
              <p>Keep word timing, language evidence and alternative candidates together.</p>
            </article>
            <article>
              <span>02</span>
              <h3>Measure risk</h3>
              <p>Combine recognition, switch-boundary, acoustic and entity evidence.</p>
            </article>
            <article>
              <span>03</span>
              <h3>Act selectively</h3>
              <p>Accept safe spans, review critical ambiguity and defer when evidence is low.</p>
            </article>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <p>SwitchSure Voice · Built for multilingual accessibility research.</p>
        <p>No raw audio persistence by default.</p>
      </footer>
    </div>
  );
}

interface EntityReviewProps {
  entity: EntityAssessment;
  index: number;
  value: string;
  confirmed: boolean;
  onChange: (index: number, value: string) => void;
  onConfirm: (index: number) => void;
  onUndo: (index: number) => void;
}

function EntityReview({
  entity,
  index,
  value,
  confirmed,
  onChange,
  onConfirm,
  onUndo,
}: EntityReviewProps) {
  const label = ENTITY_LABELS[entity.entity_type] ?? entity.entity_type.replaceAll("_", " ");
  const inputId = `entity-${index}`;
  const helpId = `${inputId}-help`;

  return (
    <li className={confirmed ? "entity-card entity-confirmed" : "entity-card"}>
      <div className="entity-card-heading">
        <span className="entity-type">{label}</span>
        <span className="entity-status">{confirmed ? "Confirmed" : "Needs your check"}</span>
      </div>
      <label htmlFor={inputId}>Check {label}</label>
      <input
        id={inputId}
        aria-describedby={helpId}
        value={value}
        onChange={(event) => onChange(index, event.target.value)}
      />
      <p id={helpId}>Keep the spoken value or type the correct one.</p>
      <div className="entity-actions">
        {confirmed ? (
          <button className="undo-button" type="button" onClick={() => onUndo(index)}>
            Undo confirmation
          </button>
        ) : (
          <button
            className="confirm-button"
            type="button"
            onClick={() => onConfirm(index)}
            disabled={!value.trim()}
          >
            Confirm {label}
          </button>
        )}
      </div>
    </li>
  );
}
