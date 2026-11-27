# Accessibility verification status

WCAG 2.2 AA is the release target. Automated checks support the review but do not establish
conformance.

## Implemented interaction contract

- Native buttons, links, textareas, labels, headings, and ordered review lists preserve keyboard
  behavior and a logical focus order.
- Submit, queued, processing, completion, cancellation, and blocking-error states are announced in
  one polite atomic live region; token updates are never streamed to assistive technology.
- Active jobs expose an explicit cancel action. Failed jobs offer a bounded retry when permitted.
- A typed transcript path remains available without microphone or audio input.
- Risk is communicated with text labels (`Clear`, `Check`, `Uncertain`) as well as color.
- Critical values require an explicit confirmation and support undo. High-stakes content is never
  auto-submitted.
- Job capabilities remain in component memory and are not written to browser storage or logs.
- Visible focus, narrow-screen layout, high-contrast-compatible controls, and reduced-motion rules
  are present in the stylesheet.

## Automated verification

React Testing Library and axe cover the successful review state, async polling, cancellation,
retry, offline recovery, typed fallback, capability non-persistence, critical-value confirmation,
undo, and safe error display.

## Manual checks still required before claiming conformance

- Keyboard-only completion in current Chrome, Firefox, and Safari.
- NVDA/Firefox and VoiceOver/Safari walkthroughs.
- 200% and 400% zoom, narrow viewport, Windows High Contrast, and reduced motion.
- Target size and error recovery review with users who have tremor or low-precision input.
- Recording permission denied, no-microphone, upload, and real slow-network flows after those media
  adapters are implemented.
