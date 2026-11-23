# ADR-0001: Python-first modular monolith

- Status: accepted
- Date: 2026-11-20

## Context

The project needs rapid research iteration, shared domain contracts, and a path from local models to GPU workers. Early microservices would add distributed failure modes before traffic or team boundaries justify them.

## Decision

Use a Python-first monorepo with a framework-independent core, FastAPI delivery layer, separate ML/evaluation packages, and an accessible TypeScript browser client. Deploy as a modular monolith initially; communicate with model adapters through explicit Python ports that can later become RPC boundaries.

## Consequences

- Research and product share typed contracts and tests.
- The browser remains capable of high-quality audio and accessibility interaction.
- Model processes can split out later without redesigning domain objects.
- Boundary discipline must be enforced in tests and reviews.
