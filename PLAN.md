# Dark Pool Detection System — Master Plan v2.0

**Autor**: Informatyk (Tech Lead)
**Data**: 2026-05-13
**Status**: 🏗️ Fazy 0-8 ukończone → Fazy 9-13 w planie

---

## 🎯 Cel

Zbudować system wykrywający aktywność w dark poolach przy użyciu:
- Publicznie dostępnych danych (FINRA ATS, TRF prints, lit market data)
- Technik ML z literatury akademickiej (VPIN, iceberg detection, order flow analysis)
- Backtestów na danych historycznych

NIE próbujemy złamać prawa ani uzyskać nieautoryzowanego dostępu — 
używamy tylko publicznych danych i opublikowanych metod.

---

## ✅ Fazy 0-8: Zrealizowane

<details>
<summary>Kliknij aby rozwinąć szczegóły zrealizowanych faz</summary>

### Faza 0: Fundamenty ✅
- Struktura projektu, config, logger, data store

### Faza 1: Data Pipeline ✅
- FINRA scraper, YFinance feed, Order Book Simulator, pipeline orchestrator

### Faza 2: Iceberg Detection ✅
- Native + synthetic iceberg detection (arXiv:1909.09495)

### Faza 3: VPIN Calculator ✅
- Volume-Synchronized PIN z BVC (Easley, López de Prado, O'Hara 2011)

### Faza 4: Dark Volume Reconstruction ✅
- TRF decomposition, FINRA ATS integration, anomaly detection

### Faza 5: Trader Type Classification ✅
- GMM klasyfikacja: Institution / HFT / Retail / Market Maker

### Faza 6: ML Prediction Layer ✅
- XGBoost Iceberg Predictor, LSTM Dark Trade Predictor (fallback XGBoost), Trader Fingerprint (KMeans/DBSCAN)

### Faza 7: Dashboard & Alerts ✅
- Streamlit dashboard, Flask web UI, Alert Engine (JSONL/CSV)

### Faza 8: Backtest & Integracja ✅
- 4 strategie (VPIN fade, Iceberg front-run, Dark volume fade, Ensemble)
- Walk-forward testing
- E2E pipeline runner

</details>

---

## 🔮 Faza 9: BACD Model — Behavioral Duration Analysis

**Inspiracja**: Repo #1 (pranay0703) — HAR-BACD-V model

### Problem
Nasz TraderTypeClassifier używa `trade_interval_mean` jako jednej cechy — to płytkie.
Tymczasem **rozkład czasów między transakcjami** jest jednym z najsilniejszych
sygnałów odróżniających instytucje od retailu w dark poolach.

Instytucje:
- Handlują w "paczkach" (bursts) — wiele transakcji w krótkim czasie, potem cisza
- Używają VWAP/TWAP algorytmów które zostawiają charakterystyczny ślad czasowy
- Mają przewidywalne interwały (algorytmiczne)

Retail:
- Handluje nieregularnie
- Pojedyncze transakcje, nie bursty
- Brak wzorca czasowego

### Implementacja

**1. BACD Feature Extractor** (`src/detection/bacd.py`)
```
Wejście: tick-level trade data
Wyjście: wektor cech duration dla każdej sesji

Cechy:
- burst_ratio: % transakcji w burstach (< 100ms od poprzedniej)
- burst_size_mean: średnia liczba transakcji w burście
- inter_burst_interval: średni czas między burstami
- duration_autocorrelation: autokorelacja interwałów (lag 1-5)
- weibull_shape: parametr kształtu rozkładu Weibulla dopasowanego do interwałów
- diurnal_pattern: odchylenie od wzorca intraday (U-shape)
- duration_dispersion: współczynnik zmienności interwałów
```

**2. BACD w TraderTypeClassifier** — rozszerzenie istniejącego klasyfikatora
o cechy duration, zwiększenie precyzji rozróżnienia Institution vs HFT

**3. Duration-based anomaly detection** — wykrywanie nietypowych wzorców czasowych
(nagłe przyspieszenie transakcji = possible iceberg execution)

### Metryki
- Trader classification precision: +10-15 p.p. (z duration features)
- Duration anomaly detection: >60% precision na symulowanych danych

---

## 🔮 Faza 10: XAI — Model Explainability

**Inspiracja**: Repo #1 — SHAP + LIME

### Problem
Nasze modele ML (XGBoost, LSTM) dają predykcje, ale nie wyjaśniają DLACZEGO.
W dark pool detection kluczowe jest zrozumienie:
- Które cechy triggerują alert?
- Dlaczego ten trade został oznaczony jako dark pool?
- Jaka jest kontrybucja VPIN vs iceberg vs volume anomaly?

### Implementacja

**1. SHAP Explainer** (`src/ml/explainability.py`)
```
- SHAP TreeExplainer dla XGBoost (IcebergPredictor)
- SHAP KernelExplainer dla fallback modeli
- Waterfall plots: które cechy pchnęły predykcję w górę/w dół
- Summary plots: globalna ważność cech
- Force plots: wyjaśnienie pojedynczej predykcji
```

**2. SHAP dla sygnałów detekcyjnych**
```
- Dekompozycja detection_score na kontrybucje: VPIN vs Iceberg vs Dark Volume
- Wyjaśnienie dlaczego overall score jest niski/wysoki
- Identyfikacja najsłabszego ogniwa w pipeline
```

**3. Raport XAI** — endpoint API `/api/explain/{prediction_id}` zwracający
feature contributions i waterfall plot jako JSON

### Metryki
- SHAP summary plots dla wszystkich 3 modeli ML
- Feature contribution endpoint w API
- Top-3 features wyjaśniające >80% wariancji predykcji

---

## 🔮 Faza 11: Bayesian Uncertainty Quantification

**Inspiracja**: Repo #1 — Bayesian NN + SNGP

### Problem
Wszystkie nasze predykcje są punktowe. Nie wiemy:
- Jak pewna jest predykcja dark trade? (80% vs 55% to duża różnica)
- Czy VPIN 0.75 to naprawdę "elevated" czy po prostu szum?
- Które alerty mają wysokie prawdopodobieństwo false positive?

W dark pool detection **fałszywe alarmy są drogie** — każdy alert wymaga ludzkiej uwagi.

### Implementacja

**1. Confidence Intervals dla predykcji** (`src/ml/uncertainty.py`)
```
- Bootstrap ensemble: N razy trenuj model na podpróbkach → rozkład predykcji
- Prediction Intervals: [dolna granica, górna granica] zamiast punktu
- Alert tylko gdy dolna granica > threshold
```

**2. VPIN Confidence Bands**
```
- Rolling VPIN volatility → adaptive threshold
- Bayesian update VPIN threshold na podstawie historycznej precyzji
- Zamiast stałego 0.8: dynamiczny threshold = baseline + 2σ
```

**3. Kalman Filter dla Dark Volume Estimate**
```
- Estymacja dark volume jako ukrytej zmiennej stanu
- Measurement: lit volume anomaly
- State transition: autoregressive dark volume process
- Output: filtered dark volume z uncertainty bounds
```

### Metryki
- 95% prediction intervals dla dark trade probability
- False positive rate ↓ 30-50% przez filtrowanie niepewnych alertów
- Kalman-filtered dark volume z confidence bands

---

## 🔮 Faza 12: Trader Fingerprint 2.0

### Problem
Nasz obecny TraderFingerprint (KMeans) działa na prostych cechach — średnie, odchylenia.
Ale **prawdziwe fingerprinty traderów** są znacznie bogatsze. Instytucje zostawiają
charakterystyczne ślady w strukturze transakcji.

### Implementacja

**1. Signature Features** (`src/detection/signature.py`)
```
Nowe cechy per-trader:
- order_aggressiveness: % market vs limit (z tick rule)
- size_consistency: CV trade sizes (instytucje = stałe rozmiary)
- venue_loyalty: % transakcji na jednym venue
- iceberg_affinity: korelacja z wykrytymi icebergami
- price_impact_persistence: jak długo trwa impact po trade'u
- round_trip_ratio: % transakcji które są częścią round-trip (buy→sell)
- cancellation_rate: stosunek anulowań do egzekucji (z order book data)
```

**2. Temporal Signature Matching**
```
- Dopasowywanie fingerprintów w czasie: "ten sam trader co 2h temu?"
- Tracking migracji traderów między venue'ami
- Detekcja nowych traderów (out-of-distribution detection)
```

**3. BACD Features w Fingerprint** (integracja z Fazą 9)
```
- duration_features jako input do klasteryzacji
- Rozróżnienie: algo (regularne interwały) vs human (nieregularne)
```

### Metryki
- Silhouette score >0.4 (obecnie ~0.1)
- Trader re-identification accuracy >60% (cross-session matching)
- ≥6 cech w fingerprint vector (obecnie 14, ale 8 to szum)

---

## 🔮 Faza 13: Advanced Detection Methods

### 13a. Quote Stuffing Detection
Wykrywanie zalewania rynku fałszywymi zleceniami (high cancel rate) —
często używane przez HFT do maskowania dark pool activity.

```
Cechy:
- cancel_to_trade_ratio > 20:1 → quote stuffing
- order_lifetime < 100ms → wash trading
- depth_volatility: nagłe zmiany głębokości order booka
```

### 13b. Layering / Spoofing Detection
Warstwowanie zleceń — składanie wielu zleceń na różnych poziomach ceny
aby stworzyć fałszywe wrażenie głębokości rynku.

```
Cechy:
- multi_level_orders: zlecenia na >3 poziomach jednocześnie
- cancel_before_execution: anulowanie zanim dojdzie do egzekucji
- price_level_oscillation: oscylacja między poziomami
```

### 13c. TRF Print Analysis
Analiza Trade Reporting Facility — opóźnione raporty z dark pool.
```
- TRF vs Exchange timing discrepancy
- Late print detection (TRF prints >10s after execution)
- Volume-weighted TRF decomposition
```

---

## 📊 Plan wdrożenia (kolejność)

```
TERAZ (Fazy 9-10):
  ├── Faza 9:  BACD Model (3-4h)
  │   ├── bacd.py — feature extraction
  │   ├── Integracja z TraderTypeClassifier
  │   └── Duration anomaly detection
  └── Faza 10: XAI Explainability (2-3h)
      ├── explainability.py — SHAP + LIME
      ├── API endpoint /api/explain
      └── Raport XAI

DALEJ (Fazy 11-12):
  ├── Faza 11: Bayesian Uncertainty (3-4h)
  │   ├── uncertainty.py — bootstrap ensemble
  │   ├── VPIN confidence bands
  │   └── Kalman filter dark volume
  └── Faza 12: Trader Fingerprint 2.0 (3-4h)
      ├── signature.py — nowe cechy
      ├── Temporal matching
      └── Integracja BACD features

POTEM (Faza 13):
  └── Faza 13: Advanced Detection (4-5h)
      ├── Quote stuffing detection
      ├── Layering / spoofing detection
      └── TRF print analysis
```

---

## 📈 Metryki sukcesu v2.0

| Moduł | Obecnie | Target po v2.0 |
|---|---|---|
| Iceberg detection precision | 91% avg confidence | >92% |
| VPIN signal usefulness | 50/100 score | >65/100 (z adaptive threshold) |
| Dark volume accuracy | 35-36% dark share | Kalman-filtered estimate |
| Trader classification | GMM 4-klastry, silhouette ~0.1 | silhouette >0.4, ≥4 wyraźne typy |
| Dark trade prediction | 80.7% accuracy (fallback) | + uncertainty bounds |
| False positive rate | Unknown | ↓ 30-50% (z Bayesian filtering) |
| Model explainability | Brak | SHAP waterfall dla każdej predykcji |
| Detection score overall | 83.3/100 | >88/100 |

---

## 🛠 Stack (bez zmian)

| Warstwa | Technologia | Uwaga |
|---|---|---|
| Język | Python 3.11 | Bez zmian |
| Detection | VPIN, Iceberg, BACD | + nowe moduły |
| ML | XGBoost, LSTM, GMM | + SHAP, Bayesian |
| Dashboard | Flask + Streamlit | + XAI endpoint |
| Backtest | vectorbt + custom | Bez zmian |
| Data | FINRA + YFinance | + TRF analysis |

---

## ⚠️ Ryzyka i ograniczenia v2.0

| Ryzyko | Mitigacja |
|---|---|
| BACD działa dobrze tylko na real data (nie synthetic) | Generować bardziej realistyczne interwały w symulatorze |
| SHAP/LIME są wolne dla dużych modeli | KernelExplainer z próbkowaniem; cache'ować wyjaśnienia |
| Bootstrap ensemble zwiększa koszt obliczeniowy 5-10x | Użyć tylko dla final prediction, nie dla każdego ticka |
| Quote stuffing detection wymaga order book data | Symulować cancellations w OrderBookSimulator |
| TRF analysis wymaga danych TRF (płatne) | Symulować TRF prints; docelowo Polygon API |
