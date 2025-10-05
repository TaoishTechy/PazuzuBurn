# PazuzuBurn v0.7 — Performance-Optimized Universal Nexus

![Status](https://img.shields.io/badge/status-research-orange)
![Version](https://img.shields.io/badge/version-v0.7-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

PazuzuBurn v0.7 is a simulation and control testbed for an optical-channel inspired plant with adaptive, safety‑aware control loops. It stress‑tests stability, neural‑coherence gating, thermal prediction, tiered risk routing, and BER‑driven throttling under controlled disturbances.

This release focuses on **performance optimization with clearer promotions/demotions, gentler CUSUM thresholds, adaptive Kalman tuning, and tier‑aware BER margins**.

---

## ✨ Highlights (v0.7)

- **Enhanced signal path**
  - `V5OpticalChannel` with refined plant matrices, thermal dynamics, and recovery mode.
  - `V5MACS_Controller` with syndrome pressure `S`, spectral abscissa tracking, thermal adaptation, and neural-aware PDM gain.
- **Adaptive estimation**
  - `KalmanStateObserver` + `InnovationAdaptiveKalman` with fixed innovation ordering and gentler covariance adaptation.
- **Safety + promotion logic**
  - `AdaptiveCreditSystem` with momentum; `WindowedPromotionVerifier` (windowed gates, relaxed criteria) and `V7RiskTieredRouter` tier gates.
  - `PILockCUSUM` detector with relaxed parameters and event counting.
- **Thermal foresight**
  - `ProactiveThermalManager` prediction horizon + cooling recommendations.
- **BER policy**
  - `BERSurrogateModel` with `BERMarginOptimizer` (tier‑aware thresholds, adaptive tighten/loosen); only throttles after consecutive violations.
- **Disturbance memory**
  - `PredictiveFailureAnalyzer` + `DisturbancePatternMemory` to learn recoveries for recurring patterns.
- **Cross-domain correlations**
  - `CrossCorrelationAnalyzer` tracks correlations among jitter, neural coherence, thermal state, syndrome, and stability.

---

## 🧩 Core Components

- **Ledger & Telemetry**
  - `MerkleLogHLA` — append‑only hash ledger for snapshots, promotions/demotions, disturbances.
- **Plant & Control**
  - `V5OpticalChannel` — state `x∈R³`, process/measurement noise, thermal proxy, recovery mode.
  - `V5MACS_Controller` — feedback gain `K`, weight matrix `W`, spectral abscissa `α`, syndrome pressure `S`, neural PDM gating, spectral adaptive gain.
- **Estimation**
  - `KalmanStateObserver`, `InnovationAdaptiveKalman`.
- **Safety & Credit**
  - `PILockCUSUM`, `AdaptiveCreditSystem`, `WindowedPromotionVerifier`, `V7RiskTieredRouter` (tier gates 5/6).
- **Performance Policies**
  - `BERMarginOptimizer`, `BERSurrogateModel`, `OTPTransitionProfiler` (pre‑ramp + notch), `ProactiveThermalManager`.
- **Analytics**
  - `StabilityMonitor` (trend + cross‑correlation), `PredictiveFailureAnalyzer`, `CrossCorrelationAnalyzer`, `DisturbancePatternMemory`.

> See the source for thresholds, window sizes, and rates used by each component.

---

## 📦 Installation

**Python 3.10+** is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy
```

No other third‑party packages are required.

---

## ▶️ Quickstart

Run the built‑in v0.7 simulation loop:

```bash
python pburn_0.7.py
```

You should see console output similar to:

```text
=== PAZUZUBURN V0.7 - PERFORMANCE OPTIMIZED UNIVERSAL NEXUS ===
Initializing optimized architecture with V0.7 enhancements...

INFO: Enhanced MerkleLogHLA initialized.
INFO: Enhanced V5OpticalChannel initialized.
INFO: KalmanStateObserver initialized.
INFO: Enhanced InnovationAdaptiveKalman initialized.
INFO: Enhanced NeuralPDMHysteresis initialized.
INFO: Enhanced AdaptiveCreditSystem initialized.
INFO: OTPTransitionProfiler initialized.
INFO: Enhanced BERMarginOptimizer initialized.
INFO: PILockCUSUM initialized.
INFO: Enhanced WindowedPromotionVerifier initialized.
INFO: V7RiskTieredRouter initialized.
Initial Ledger Root: <hex>...

--- Step 1/60 [Tier 4] ---
  METRICS: Jitter=... | Temp=...°C | Stability=...
  CONTROL: S=... | α=... | PDM=... | Kalman_Adapt=...
  CREDITS: T5=.../3.0 | T6=.../5.0 | DemotionWatch=...
```

---

## 🧪 What the loop does

- Steps a 3‑state plant with periodic "write" cycles and thermal accumulation/cooling.
- Estimates state with an adaptive Kalman filter.
- Computes syndrome pressure `S`, stability score, and BER risk; applies throttling only on **persistent** risk.
- Predicts thermal trajectory and applies **cooling boost** recommendations.
- Uses neural‑coherence gating to **defer writes** during low coherence (with derivative‑aware exceptions).
- Logs snapshots, promotions/demotions, and controlled disturbances to the Merkle ledger.

---

## ⚙️ Tiers, Credits, and Promotion

- **Credits:** tier‑scoped safety credits (`T5`, `T6`) accumulate based on performance index (jitter/thermal/stability/syndrome).
- **Promotion:** `WindowedPromotionVerifier` checks rolling gates (jitter, neural coherence, stability) and allows promotion when enough passes (or strong 2‑of‑3 performance).
- **Demotion:** triggers when credits fall under demotion thresholds for several consecutive checks.

---

## 🧊 Thermal & OTP Pre‑Ramp

- `ProactiveThermalManager` predicts the next few steps and suggests cooling boosts (low/medium/high).
- `OTPTransitionProfiler` predicts OTP‑like transitions from write patterns and applies a **pre‑ramp** and **adaptive notch** to reduce shock.

---

## 📈 Metrics & Reports

At the end of a run, the simulation prints a summary with average stability, jitter, neural coherence, Kalman adaptation count, CUSUM detections, and any cross‑domain correlation insights.

---

## 🧩 Import as a Library

You can also import and run the simulation from Python:

```python
from pburn_0.7 import run_v7_simulation

run_v7_simulation()
```

> Classes like `V5OpticalChannel`, `V5MACS_Controller`, `AdaptiveCreditSystem`, etc., can be imported and reused independently for experiments.

---

## 🗂️ Project Structure (single‑file)

```
pburn_0.7.py
├─ MerkleLogHLA
├─ StabilityMonitor, CrossCorrelationAnalyzer, DisturbancePatternMemory
├─ PredictiveFailureAnalyzer
├─ PILockCUSUM
├─ NeuralPDMHysteresis, NeuralAwarePDM, SpectralAdaptiveGain
├─ AdaptiveCreditSystem, WindowedPromotionVerifier
├─ KalmanStateObserver, InnovationAdaptiveKalman
├─ BERMarginOptimizer, BERSurrogateModel
├─ ProactiveThermalManager
├─ OTPTransitionProfiler
├─ V5OpticalChannel
├─ V5MACS_Controller
└─ run_v7_simulation()
```

---

## 🧭 Roadmap

- Configurable CLI flags (steps, seeds, disturbance schedule, thresholds).
- Structured JSON logging for snapshots and final reports.
- Plotting utilities for trends (stability, neural coherence, credits, BER risk).
- Realtime dashboard (optional) for interactive runs.

---

## 🤝 Contributing

Pull requests welcome. Please include clear descriptions and testable changes.

---

## 📝 License

MIT — see `LICENSE` (add one if missing).

---

## Acknowledgments

This work draws on control theory, adaptive estimation, and reliability‑oriented design patterns. Special thanks to contributors experimenting with cross‑domain correlation tracking and neural‑coherence gating.

