<!-- ═══════════════════════════════════════════════════════════════════ -->
<!--  GitHub will render this beautifully.                                 -->
<!-- ═══════════════════════════════════════════════════════════════════ -->

<p align="center">
  <h1 align="center">🌑 Dark Pool Detection System</h1>
  <p align="center">
    <em>Detecting Hidden Liquidity and Informed Trading in Off-Exchange Venues<br>
    Using Public Data, Microstructure Theory, and Machine Learning</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/tests-179%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/modules-35%20files-informational" alt="Modules">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## Abstract

We present a modular, open-source system for detecting dark pool trading activity
in U.S. equity markets. The system combines four detection methodologies drawn from
the market microstructure literature — Volume-Synchronized Probability of Informed
Trading (VPIN), iceberg order detection, dark volume reconstruction from TRF/FINRA
data, and trader type classification via Gaussian Mixture Models — with a machine
learning prediction layer (XGBoost, LSTM, K-Means clustering). A behavioral duration
analysis module (BACD) extends the trader fingerprint with temporal signatures.
All components are integrated into an end-to-end pipeline with a Flask web interface,
REST API, multi-level alert engine, and strategy backtesting framework. The system
uses only publicly available data (FINRA ATS, Yahoo Finance) and published academic
methods, requiring no paid APIs.

**Keywords**: dark pool, VPIN, iceberg orders, market microstructure, XGBoost,
Gaussian Mixture Models, order flow toxicity, trade classification, BACD, SHAP

---

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Architecture](#2-architecture)
- [3. Detection Methodology](#3-detection-methodology)
  - [3.1 VPIN — Volume-Synchronized PIN](#31-vpin--volume-synchronized-pin)
  - [3.2 Iceberg Order Detection](#32-iceberg-order-detection)
  - [3.3 Dark Volume Reconstruction](#33-dark-volume-reconstruction)
  - [3.4 Trader Type Classification](#34-trader-type-classification)
  - [3.5 BACD — Behavioral Duration Analysis](#35-bacd--behavioral-duration-analysis)
- [4. Machine Learning Layer](#4-machine-learning-layer)
- [5. Explainability (XAI)](#5-explainability-xai)
- [6. Installation](#6-installation)
- [7. Quick Start](#7-quick-start)
- [8. Web Interface & API](#8-web-interface--api)
- [9. Alert System](#9-alert-system)
- [10. Backtesting](#10-backtesting)
- [11. Test Suite](#11-test-suite)
- [12. Results](#12-results)
- [13. Competitive Analysis](#13-competitive-analysis)
- [14. Roadmap](#14-roadmap)
- [15. References](#15-references)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. Introduction

### 1.1 Motivation

Dark pools — alternative trading systems that do not display quotes publicly —
account for approximately 35–40% of U.S. equity trading volume (SEC, 2022).
Despite their scale, dark pool activity remains largely opaque to regulators,
researchers, and market participants. Detecting and characterizing this hidden
liquidity is critical for:

- **Market surveillance**: identifying potential manipulation (quote stuffing,
  layering, spoofing) that exploits dark venues
- **Transaction cost analysis**: understanding where institutional orders are
  actually executed
- **Alpha research**: dark pool prints often precede significant price moves
- **Regulatory compliance**: FINRA Rule 4552 requires ATS transparency reporting

### 1.2 Design Principles

| Principle | Implementation |
|---|---|
| **Public data only** | FINRA ATS weekly reports + Yahoo Finance + simulated order books |
| **Published methods** | VPIN (Easley et al., 2011), iceberg detection (Zotikov, 2019) |
| **Reproducible** | Deterministic seeds, CI/CD pipeline, 179 automated tests |
| **Self-contained** | Single `pip install`, Docker one-command launch |
| **Explainable** | SHAP/LIME for ML decisions, score decomposition |

### 1.3 Comparison with Existing Work

A systematic review of open-source dark pool projects found that existing
repositories either provide sophisticated ML with no infrastructure, or
production-ready streaming with trivial detection logic (single volume
thresholds). This system is, to our knowledge, the only open-source project
combining academic detection methods, ML prediction, web interface, REST API,
alert engine, backtesting, Docker deployment, and CI/CD in a single codebase
(see [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md) for full audit).

---

## 2. Architecture

```
                                ┌──────────────────────┐
                                │    DATA PIPELINE      │
                                ├──────────────────────┤
                                │  FINRA ATS Scraper   │──► Weekly dark pool
                                │  YFinance Feed       │──► Lit market OHLCV
                                │  OrderBook Simulator │──► Synthetic ticks
                                └──────────┬───────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │  ICEBERG DETECTOR │  │  VPIN CALCULATOR  │  │  DARK VOLUME RECON │
        │  native+synthetic  │  │  BVC + tick rule  │  │  TRF decomposition │
        └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │ TRADER CLASSIFIER │  │  BACD ANALYZER    │  │ TRADER FINGERPRINT│
        │ GMM 4-cluster     │  │  Weibull + bursts  │  │ KMeans/DBSCAN     │
        └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘
                  │                      │                      │
                  └──────────────────────┼──────────────────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          │      ML PREDICTION LAYER     │
                          ├─────────────────────────────┤
                          │  XGBoost  → Iceberg Fill    │
                          │  LSTM     → Dark Trade      │
                          │  XAI/SHAP → Explainability  │
                          └──────────────┬──────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
        ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐
        │  FLASK WEB UI     │  │  ALERT ENGINE      │  │  BACKTEST ENGINE  │
        │  5 pages + REST   │  │  JSONL/CSV/Webhook │  │  4 strategies     │
        └───────────────────┘  └───────────────────┘  └───────────────────┘
```

### Project Structure

```
dark-pool-detection/
├── src/
│   ├── data/               # Data Pipeline (Faza 1)
│   │   ├── finra_scraper.py      FINRA ATS weekly reports
│   │   ├── yfinance_feed.py      Lit market OHLCV
│   │   └── simulator.py          Order book simulator (GBM + agents)
│   ├── detection/           # Detection Engine (Fazy 2–5, 9)
│   │   ├── iceberg.py            Iceberg order detection
│   │   ├── vpin.py               VPIN calculator (ELO 2011)
│   │   ├── dark_volume.py        Dark volume reconstruction
│   │   ├── trader_type.py        Trader type classification
│   │   └── bacd.py               Behavioral duration analysis
│   ├── ml/                  # ML Layer (Fazy 6, 10)
│   │   ├── iceberg_predictor.py  XGBoost fill predictor
│   │   ├── dark_trade_predictor.py LSTM/XGBoost fallback
│   │   ├── trader_fingerprint.py KMeans/DBSCAN clustering
│   │   └── explainability.py     SHAP + QuickExplainer
│   ├── alerts/engine.py     # Alert Engine (Faza 7a)
│   ├── dashboard/app.py     # Streamlit Dashboard (Faza 7b)
│   ├── backtest/engine.py   # Backtesting (Faza 8a)
│   ├── web/                 # Flask Web Interface
│   │   ├── app.py                 Flask app + REST API
│   │   ├── templates/             HTML templates (5 pages)
│   │   └── static/style.css       Dark theme CSS
│   ├── pipeline.py          # Main integration pipeline
│   └── utils.py             # Config, logging
├── pipelines/run_all.py     # End-to-end runner
├── tests/                   # 179 unit + integration tests
├── config/                  # YAML configuration
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # 1-service orchestration
├── .github/workflows/       # CI/CD (tests + Docker push)
├── PLAN.md                  # Master plan v2.0
├── COMPETITIVE_ANALYSIS.md  # Deep audit of existing repos
└── README.md                # This file
```

---

## 3. Detection Methodology

### 3.1 VPIN — Volume-Synchronized PIN

**Reference**: Easley, D., López de Prado, M., & O'Hara, M. (2011).
*"Volume-Synchronized Probability of Informed Trading."*

VPIN measures order flow toxicity by computing the imbalance between buy and
sell volume within volume-synchronized buckets (rather than fixed-time bars).
High VPIN values (>0.8) indicate that an informed trader is actively trading
against uninformed flow — a signal often associated with dark pool activity.

**Implementation**:
- Bulk Volume Classification (BVC) with z-score-based buy probability
- Tick rule fallback when BVC is unreliable
- Rolling VPIN over configurable number of buckets (default: 50)
- Real-time streaming `update()` method for tick-by-tick processing

### 3.2 Iceberg Order Detection

**Reference**: Zotikov, A. (2019). *"CME Iceberg Order Detection and
Prediction."* arXiv:1909.09495

| Method | Signature | Confidence |
|---|---|---|
| **Native** | Order book volume does NOT decrease after trade → hidden reserve | `discrepancy = resting_vol / trade_size` |
| **Synthetic** | Identical-size limit orders arrive within 50ms of each trade | `confidence = 0.5 + (n_repeated - 1) × 0.15` |

**Size estimation**: Kaplan-Meier survival estimator on partial fills.

### 3.3 Dark Volume Reconstruction

Dark pool volume is estimated as the residual between total reported volume
(TRF prints) and lit exchange volume:

```
dark_est = base_dark × anomaly_mult × time_of_day_mult
```
- `base_dark = lit_vol × 0.35/0.65` — industry baseline
- `anomaly_mult` amplified during volume spikes (|z| > 3σ)
- `time_of_day_mult = 1.3` during open/close

### 3.4 Trader Type Classification

GMM with 19 features (10 base + 9 BACD) → 4 clusters:

| Type | Size | Interval | Dark | BACD Signature |
|---|---|---|---|---|
| **Institution** | Large | Moderate | High | Bursty, autocorrelated |
| **HFT** | Small | Ultra-fast | Low | Extreme bursts, strong AC |
| **Retail** | Very small | Irregular | None | No bursts, no AC |
| **Market Maker** | Medium | Fast | Neutral | Negative AC (mean-reverting) |

### 3.5 BACD — Behavioral Duration Analysis

**Reference**: Engle & Russell (1998). *"Autoregressive Conditional Duration."* Econometrica.

The time *between* trades carries as much information as the trades themselves.

**Features**: burst_ratio, burst_size_mean, Weibull shape (R²>0.9), duration
autocorrelation, diurnal deviation, burstiness_index, duration CV.

---

## 4. Machine Learning Layer

| Model | Task | Architecture | Performance |
|---|---|---|---|
| **Iceberg Predictor** | Will iceberg fill within N trades? | XGBoost, 10 features, time-series CV | 49.4% CV acc |
| **Dark Trade Predictor** | Will dark trade occur in K seconds? | LSTM 2×64 / XGBoost fallback | 80.7% acc |
| **Trader Fingerprint** | Identify trading entities | KMeans / DBSCAN + PCA + t-SNE | Silhouette 0.10–0.46 |

---

## 5. Explainability (XAI)

- **SHAP TreeExplainer**: Exact SHAP values for XGBoost (waterfall, summary, force plots)
- **QuickExplainer**: SHAP-free fallback with permutation importance + feature perturbation
- **Score decomposition**: `GET /api/explain` → VPIN vs Iceberg vs Dark contribution

---

## 6. Installation

```bash
# Option A: pip
git clone https://github.com/sirfragles/dark-pool-detection.git
cd dark-pool-detection
pip install -r requirements.txt

# Option B: Docker (recommended)
docker pull ghcr.io/sirfragles/dark-pool-detection:latest
docker run -p 5000:5000 ghcr.io/sirfragles/dark-pool-detection:latest

# Option C: Docker Compose
docker compose up -d
```

---

## 7. Quick Start

```bash
# CLI: full pipeline
python3 pipelines/run_all.py

# CLI: live data
python3 pipelines/run_all.py --mode live --tickers SPY QQQ AAPL

# Web interface
python3 -m src.web.app --port 5000
# → http://localhost:5000

# Streamlit dashboard
streamlit run src/dashboard/app.py
# → http://localhost:8501

# Python API
from src.pipeline import DarkPoolPipeline
results = DarkPoolPipeline().run_simulation(n_ticks=5000, n_tickers=5, seed=42)
print(f"Score: {results['detection_score']['overall']:.1f}/100")
```

```bash
make test          # 179 tests
make run-web       # Flask UI
make run-pipeline  # Detection pipeline
make docker-build  # Docker image
```

---

## 8. Web Interface & API

| Page | Route | Description |
|---|---|---|
| Dashboard | `/` | Run pipeline, view KPI cards |
| Results | `/results` | Full report with score breakdown |
| Alerts | `/alerts` | Alert history |
| Live Data | `/live` | YFinance + dark volume estimate |

**REST API**: `GET /api/results`, `/api/health`, `/api/alerts`, `/api/vpin`,
`/api/duration`, `/api/explain`, `/api/report` · `POST /run`, `/api/simulate`,
`/api/live`

---

## 9. Alert System

| Level | Trigger |
|---|---|
| 🚨 **CRITICAL** | VPIN > 0.8, dark trade prob > 0.7 |
| ⚠️ **WARNING** | VPIN > 0.6, dark share > 40% |
| 👀 **WATCH** | Hidden volume > 5k, volume anomalies |
| ℹ️ **INFO** | Active icebergs detected |

Output: JSONL (append-only), CSV export, custom handler system.

---

## 10. Backtesting

| Strategy | Hypothesis |
|---|---|
| **VPIN Fade** | High VPIN → overreaction → mean reversion |
| **Iceberg Front-Run** | Iceberg = large buyer → price pressure |
| **Dark Volume Fade** | Volume spike → temporary dislocation |
| **Ensemble** | Weighted combination of all signals |

Walk-forward out-of-sample testing with time-series cross-validation.

---

## 11. Test Suite

```
============================= 179 passed in 14.21s ==============================
```

| File | Scope | Tests |
|---|---|---|
| `test_data.py` | OrderBookSimulator, YFinanceFeed | 21 |
| `test_detection.py` | Iceberg, VPIN, Dark Volume, Trader Type | 35 |
| `test_ml.py` | Iceberg Predictor, Dark Trade, Fingerprint | 20 |
| `test_alerts_backtest.py` | Alert Engine, Backtest Engine | 32 |
| `test_integration.py` | Pipeline E2E, cross-module, performance | 18 |
| `test_bacd_xai.py` | BACD Analyzer, XAI Explainability | 22 |

---

## 12. Results

```
╔══════════════════════════════════════════════════════════╗
║         DARK POOL DETECTION — PIPELINE REPORT            ║
╠══════════════════════════════════════════════════════════╣
║  📊 SIMULATION                                           ║
║     Trades total:        4,180                            ║
║     Dark pool:             963 (23.0%)                    ║
║     Iceberg:             1,805 (43.2%)                    ║
║  🧊 ICEBERG DETECTION                                     ║
║     Active icebergs:     2,930                            ║
║     Est. hidden vol:  1,354,558 shares                    ║
║     Avg confidence:     91.58%                            ║
║  ⚡ VPIN                                                   ║
║     Mean:                0.329                            ║
║  📈 DETECTION QUALITY                                      ║
║     Iceberg:  100.0   VPIN:  50.0   Dark:  99.0           ║
║     OVERALL:              83.0/100                        ║
╚══════════════════════════════════════════════════════════╝
```

| Operation | Time |
|---|---|
| BACD (2000 trades) | 0.007s |
| Full pipeline (5×5000) | 4.3s |

---

## 13. Competitive Analysis

Full audit in [COMPETITIVE_ANALYSIS.md](COMPETITIVE_ANALYSIS.md).

| Feature | Here | #1 | #2 | #3 |
|---|---|---|---|---|
| VPIN (ELO 2011) | ✅ | ❌ | ❌ | ❌ |
| Iceberg (Zotikov 2019) | ✅ | ❌ | ❌ | ❌ |
| BACD duration | ✅ | ✅ | ❌ | ❌ |
| XAI explainability | ✅ | ✅ | ❌ | ❌ |
| Backtesting | ✅ | ❌ | ❌ | ❌ |
| Web UI + API | ✅ | ❌ | ✅ | ❌ |
| Docker + CI/CD | ✅ | ❌ | ✅ | ❌ |
| Test suite | ✅ 179 | ❌ | ❌ | ❌ |
| Public data | ✅ | ❌ | ❌ | ❌ |

---

## 14. Roadmap

| Phase | Module | Status |
|---|---|---|
| 0–8 | Core system | ✅ Complete |
| 9 | BACD | ✅ Complete |
| 10 | XAI | ✅ Complete |
| 11 | Bayesian uncertainty | 🔮 Planned |
| 12 | Fingerprint 2.0 | 🔮 Planned |
| 13 | Advanced detection | 🔮 Planned |

---

## 15. References

1. Easley, López de Prado, O'Hara (2011). *VPIN.* J. Financial Economics.
2. Zotikov (2019). *CME Iceberg Detection.* arXiv:1909.09495.
3. Easley, López de Prado, O'Hara (2012). *Flow Toxicity.* Rev. Financial Studies.
4. Engle & Russell (1998). *Autoregressive Conditional Duration.* Econometrica.
5. Lundberg & Lee (2017). *SHAP.* NeurIPS.
6. Ribeiro, Singh & Guestrin (2016). *LIME.* KDD.
7. FINRA Rule 4552 — ATS Transparency.
8. Chen & Guestrin (2016). *XGBoost.* KDD.

---

## 16. License & Disclaimer

MIT License.

> **Research Use Only**: This system is an academic research tool. Results are
> for educational and analytical purposes only. No investment advice is provided.
> Dark pool volume estimates are approximations based on public data and published
> heuristics. The authors assume no liability for trading decisions made using
> this software.

---

<p align="center">
  <sub>Built with Python · XGBoost · PyTorch · Flask · Streamlit · Docker</sub>
  <br>
  <sub>© 2026 — Dark Pool Detection System v0.2.0</sub>
</p>
