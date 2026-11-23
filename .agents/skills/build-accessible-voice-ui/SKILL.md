---
name: build-accessible-voice-ui
description: Build and review the SwitchSure React/TypeScript voice client. Use when changing audio recording or upload, transcript and uncertainty presentation, entity confirmation, typed fallback, responsive UI, accessibility behavior, or browser tests under apps/web.
---

# Build the accessible voice UI

## Start with the product contract

Read `../../../docs/product/product-brief.md` and the web-client section of `../../../docs/architecture/system-design.md`. Read `references/accessibility-checklist.md` before changing an interaction.

## Workflow

1. Describe the user's task, success state, failure states, and recovery path.
2. Extend typed API/domain contracts before inventing component-local shapes.
3. Build semantic HTML and native controls first. Add ARIA only where native semantics are insufficient.
4. Keep recording, upload, and typed input as equivalent paths. Handle denied permission, no microphone, unsupported media, cancellation, timeout, offline, and retry states.
5. Present uncertainty as an action: accepted, needs review, or needs repeat. Do not expose raw probabilities as truth.
6. Preserve critical entity surface text and show alternatives/provenance in a focused review queue. Never silently replace a confirmed value.
7. Test behavior, keyboard operation, focus, live announcements, error association, contrast, reduced motion, narrow screens, and slow responses.

## Interaction rules

- Do not rely on color, hover, audio, or animation alone.
- Announce recording and processing state without repeatedly announcing token updates.
- Move focus only after an explicit user action or when required to reach a blocking error.
- Make the smallest meaningful review unit keyboard addressable; keep token-level diagnostics out of the default screen-reader path.
- Provide undo for confirmations and edits.
- Use plain language such as “Please check this medicine name,” not calibration jargon.
- Do not auto-submit high-stakes content after transcription.

## Engineering rules

- Use strict TypeScript and generated or shared API types.
- Isolate browser media APIs behind a small adapter.
- Keep server state, recording state, and review state distinct.
- Use React Testing Library for user behavior, axe for automated accessibility checks, and Playwright for critical browser flows.
- Prefer CSS tokens and visible focus styles. Respect `prefers-reduced-motion`.
- Do not log transcript, audio, entity values, or object URLs.

## Done criteria

Confirm normal, empty, loading, partial, error, retry, and permission-denied states. Verify keyboard-only completion and the typed fallback. Add tests for every new state and update the accessibility checklist when the interaction contract changes.
