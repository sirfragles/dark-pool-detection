# 🌑 Dark Pool Detection System

**Wykrywanie aktywności w dark poolach z użyciem publicznych danych i ML**

---

## 🚀 Quick Start

```bash
# Lokalnie
pip install -r requirements.txt
python3 -m src.web.app --port 5000
# → http://localhost:5000

# Docker
./docker-run.sh build && ./docker-run.sh up
# → http://localhost:5000

# Docker Hub (pre-built)
docker pull sirfragles/dark-pool-detection:latest
docker run -p 5000:5000 sirfragles/dark-pool-detection:latest
```

## 🧪 Tests

```bash
make test        # 157 testów
make test-cov    # z coverage
```

## 📊 Features

| Moduł | Opis | Źródło |
|---|---|---|
| 🧊 Iceberg Detection | Hidden order detection (native + synthetic) | arXiv:1909.09495 |
| ⚡ VPIN | Volume-Synchronized PIN | ELO (2011) |
| 🌑 Dark Volume | Dark pool reconstruction | FINRA ATS |
| 👥 Trader Type | Institution/HFT/Retail/MM classification | GMM clustering |
| 🤖 ML | XGBoost iceberg predict · LSTM dark trade · Fingerprint clustering | — |
| 📊 Web UI | Flask dashboard (4 pages) + REST API | — |
| 🚨 Alerts | Multi-level alert engine (JSONL/CSV) | — |
| 📈 Backtest | VPIN fade · Iceberg front-run · Ensemble | — |

## 🐳 Docker

```bash
# Build & run
./docker-run.sh build
./docker-run.sh up

# Or use pre-built image from Docker Hub
docker pull sirfragles/dark-pool-detection:latest
```

GitHub Actions automatically builds and pushes to Docker Hub on every push.
