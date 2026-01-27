# Testing Strategy

This document outlines the testing approach for the Model Benchmark Explorer. Use this as a guide for implementing new tests and maintaining high reliability.

## 🎯 Objectives

- **Core Stability**: Protect critical flows (scanning, and analysis).
- **Regression Prevention**: Catch bugs before they reach the user.
- **Fairness Assurance**: Ensure metrics are calculated consistently across all models.

## 🏗 Testing Levels

### 1. Backend (Python)

- **Unit Tests (Pytest)**:
  - **Location**: `backend/tests/unit` (Recommended)
  - **Focus**: Pure logic in `app/services/` (Prompt ordering, metadata parsing) and Utilities.
  - **Mocks**: Use `unittest.mock` to bypass actual ML model loading and database I/O.
- **Integration Tests**:
  - **Location**: `backend/tests/integration`
  - **Focus**: API routes in `app/api/` using FastAPI's `TestClient`. Verify DB interactions and status codes.
- **ML Verification**:
  - **Focus**: Small, deterministic datasets to verify that `analysis.py` returns correct CLIP/LPIPS scores relative to expected values.

### 2. Frontend (TypeScript)

- **Unit/Component Tests (Vitest + React Testing Library)**:
  - **Focus**: Complex UI logic and heavy helper functions.
  - **Critical Components**: `ScatterPlot`, `TransferList`, and metric calculation math.
- **End-to-End (E2E) Tests (Playwright)**:
  - **Focus**: High-level user journeys.
  - **Scenarios**:
    - "Select models from Library -> Add to Queue -> Start Benchmark".
    - "Navigate to Gallery -> Filter by Seed -> Expand Image".
    - "Switch Data Source in Dashboard -> Verify Chart Updates".

## 🛡 Critical Coverage Areas

| Area                         | Priority    | Testing Level    |
| :--------------------------- | :---------- | :--------------- |
| **Model Scanning & Hashing** | 🔥 Critical | Unit/Integration |
| **Prompt Paired/Text Logic** | 🔥 Critical | Unit             |
| **Metric Aggregation**       | ⚡ High     | Unit             |
| **Chart Data Filtering**     | ⚡ High     | Unit             |
| **Model Selection (DND)**    | 🟢 Medium   | E2E              |

## 🛠 Toolset

- **Backend**: `pytest`, `pytest-mock`, `FastAPI TestClient`.
- **Frontend**: `vitest`, `playwright`.

## 📝 Guidelines for Agents

- **Don't touch the hardware**: Always mock GPU-bound tasks (Diffusers, Torch) in local unit tests.
- **Use deterministic seeds**: Ensure all test-related benchmarking uses fixed seeds.
- **Clean state**: Always use a temporary SQLite database (memory) or clean assets directory for integration tests.
