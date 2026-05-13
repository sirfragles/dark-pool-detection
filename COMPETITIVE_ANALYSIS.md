# 🔬 Dark Pool Detection — Deep Technical Audit

Dogłębna analiza kodu źródłowego 3 repozytoriów (pomijam #4 krypto).
Data: 2026-05-13

---

## 1. pranay0703/dark-pool-fraud-detection — GŁĘBOKA ANALIZA KODU

### Co NAPRAWDĘ jest w środku (przeczytałem kod źródłowy)

```
src/models/
├── temporal_gnn.py          (13.6 KB)  — TGNN z GAT + Memory Module
├── transformer_model.py     (13.4 KB)  — Transformer z positional encoding
├── hybrid_model.py          (14.5 KB)  — HAR-BACD-V: Heterogeneous Autoregressive + 
│                                         Behavioral Conditional Duration
├── integrated_model.py      (17.1 KB)  — Ensemble TGNN + Transformer + HAR-BACD
├── explainability.py        (17.2 KB)  — SHAP + LIME (XAI)
└── uncertainty_quantification.py (15.9 KB) — Bayesian NN + SNGP

src/data_pipeline/
├── temporal_graph.py        (11.7 KB)  — Event-Based Temporal Graph construction
└── data_loader.py           (9.4 KB)   — Data loading + preprocessing

src/training/
└── trainer.py               (25.4 KB)  — Full training loop z augmentacją
```

### Co robi LEPIEJ niż my

#### 1. Temporal Graph Neural Network z GAT (Graph Attention)
```python
# Ich kod — prawdziwy TGNN z mechanizmem pamięci:
class MemoryModule(nn.Module):
    def forward(self, node_features, memory, update_vector):
        # GRU-style memory: update_gate * old + (1-update_gate) * new
        update_gate = self.update_net(combined)
        new_memory = self.new_net(combined)
        return update_gate * memory + (1 - update_gate) * new_memory

class TemporalGraphLayer(nn.Module):
    def __init__(self, ...):
        self.gat = GATConv(in_channels=..., heads=8)  # Multi-head attention
        self.memory = MemoryModule(...)  # Persistent memory across time
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.residual_proj = nn.Linear(...)  # Residual connection
```
**Nasza przewaga tutaj**: Mamy Iceberg Detection + VPIN z literatury — oni wykrywają FRAUD (inne zadanie). Ich TGNN jest świetny do relacji trader-trader, nie do samej detekcji dark pool.

#### 2. HAR-BACD-V Hybrid Model (Multi-Scale)
```python
class HeterogeneousAutoregressive(nn.Module):
    # HAR model z 3 skalami czasowymi: daily (1), weekly (5), monthly (22)
    har_lags = [1, 5, 22]
    # Każda skala ma własną sieć → potem agregacja

class BehavioralAutoregressiveConditionalDuration(nn.Module):
    # BACD model — modelowanie czasu MIĘDZY transakcjami
    # To jest kluczowe dla dark pool! Instytucje mają inne interwały niż retail.
```
**To jest coś czego NIE MAMY** — modelowanie duration między transakcjami. Nasz TraderTypeClassifier robi to powierzchownie przez `trade_interval_mean`. BACD jest specjalizowanym modelem ekonometrycznym do tego.

#### 3. XAI (SHAP + LIME) — Explainability
```python
class ModelExplainer:
    def setup_shap_explainer(self, background_data, explainer_type='kernel'):
        # Obsługa 3 typów: KernelExplainer, DeepExplainer, GradientExplainer
```
**Tego u nas BRAK** — pokazujemy feature importance z XGBoost, ale nie mamy pełnego SHAP/LIME.

#### 4. Uncertainty Quantification (Bayesian NN + SNGP)
```python
class BayesianLinear(nn.Module):
    # Wagi jako rozkłady prawdopodobieństwa (μ, log σ²)
    # Sample'owanie z posterior → przedziały ufności dla predykcji
    def forward(self, x, sample=True):
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
```
**Tego u nas BRAK** — nasze predykcje są punktowe bez przedziałów ufności. W dark pool detection uncertainty jest kluczowe (fałszywe alarmy są drogie).

### Czego im BRAKUJE (nasza przewaga)

| Czego nie mają | My mamy |
|---|---|
| ❌ Prawdziwe dane rynkowe | ✅ FINRA ATS + YFinance live |
| ❌ Docker / CI/CD | ✅ Pełny pipeline |
| ❌ Web UI / API | ✅ Flask + Streamlit + REST |
| ❌ Testy (0) | ✅ 157 testów |
| ❌ VPIN / Iceberg detection | ✅ Obie metody z literatury |
| ❌ Backtest | ✅ 4 strategie |
| ❌ Alert system | ✅ JSONL/CSV |
| ❌ Dokumentacja (poza README) | ✅ 17 KB README + PLAN.md |
| ❌ Detekcja aktywności dark pool | ❌ Oni wykrywają FRAUD, nie samą aktywność |

### Werdykt: **ML jest lepsze, produkt gorszy**

Ich ML jest 2 poziomy wyżej (TGNN + Transformer + Bayesian NN + XAI vs nasz XGBoost + LSTM + k-Means). Ale nie mają ŻADNEJ infrastruktury produkcyjnej. To projekt badawczy, nie system.

**Co przejąć**: BACD model do TraderTypeClassifier, Bayesian uncertainty do predykcji, SHAP do explainability.

---

## 2. sagarvrma/darkpooldetector — GŁĘBOKA ANALIZA KODU

### Co NAPRAWDĘ jest w środku

```
docker-compose.yml (1.9 KB):
  ├── zookeeper       — Confluent 7.5.0
  ├── kafka           — Confluent 7.5.0
  ├── kafka-ui        — Dashboard do monitorowania Kafki
  ├── spark-master    — Apache Spark 3.5.1
  ├── spark-worker    — 4 GB RAM, 4 cores
  └── fastapi          — Python 3.11-slim + FastAPI + uvicorn

data-ingestion/producer.py (2.4 KB):
  — Losowy generator trade'ów z 5% szansą na "BLOCK" trade
  — Wysyła do Kafka topic 'market-trades'

spark-jobs/detect_dark_pool.py (2.9 KB):
  — Spark Structured Streaming czyta z Kafki
  — 30-sekundowe okna sliding window
  — Klasyfikacja: vol > 50000 = DARK_POOL_ALERT, vol > 20000 = SUSPICIOUS
  — Wysyła alerty do Kafka topic 'dark-pool-alerts'

api/main.py (2.6 KB):
  — FastAPI z WebSocket endpointem
  — Streamuje alerty do frontendu
```

### Brutalna prawda o kodzie

**Spark job "detect_dark_pool.py" — CAŁA logika detekcji:**
```python
alerts = windowed.withColumn(
    "signal",
    when(col("total_volume") > 50000, "DARK_POOL_ALERT")
    .when(col("total_volume") > 20000, "SUSPICIOUS")
    .otherwise("NORMAL")
)
```

To jest **pojedynczy próg wolumenowy**. Żadnego ML. Żadnego VPIN. Żadnej analizy order flow. Po prostu: "jeśli volume > 50k → dark pool alert".

**Producer — CAŁA logika danych:**
```python
def generate_trade():
    is_dark_pool_signal = random.random() < 0.05  # 5% szansy
    if is_dark_pool_signal:
        volume = random.randint(base_vol * 10, base_vol * 50)  # 10x-50x spike
```

Dane są **całkowicie losowe** — nie modelują żadnej rzeczywistej dynamiki rynkowej. Nasz OrderBookSimulator z GBM + informed trader agent + iceberg state machine jest o wiele bardziej realistyczny.

### Co robi LEPIEJ niż my

| Ich przewaga | Szczegóły |
|---|---|
| **Real-time streaming** | Kafka + Spark Structured Streaming — prawdziwy pipeline event-driven |
| **Skalowalność** | Spark worker z 4 GB RAM / 4 cores — może przetwarzać miliony eventów |
| **WebSocket push** | FastAPI WebSocket — alerty w czasie rzeczywistym do frontendu |
| **Infrastruktura** | 7-serwisowy Docker Compose (prod-ready) |

### Co robi GORZEJ niż my

| Nasza przewaga | Szczegóły |
|---|---|
| **ML / detection** | Ich "detekcja" to jeden if-statement. My mamy VPIN + iceberg + ML ensemble. |
| **Dane** | Ich dane są losowe. Nasz symulator ma GBM + strategie traderów. |
| **Metody akademickie** | Zero. My: ELO (2011), arXiv:1909.09495. |
| **Backtest** | Brak. My: 4 strategie + walk-forward. |
| **Testy** | Zero. My: 157. |
| **Real market data** | Brak. My: FINRA ATS + YFinance live. |

### Werdykt: **Infrastruktura świetna, logika detekcji żenująca**

Najlepsza architektura streamingowa ze wszystkich 4 repo — Kafka + Spark + FastAPI + React to stack produkcyjny. Ale sama "detekcja" to jeden twardy threshold na wolumenie. To tak jakby zbudować Fabrykę Tesla żeby produkować pinezki.

**Co przejąć**: Kafka + Spark streaming pipeline, WebSocket push alerts, React dashboard jako Faza 9.

---

## 3. stefluhh/realtime-stock-exchange-analysis — GŁĘBOKA ANALIZA

### Co NAPRAWDĘ jest w środku

```
Kotlin / Spring Boot / Maven:
├── StockpriceStreamingAdapter.kt  — Polygon.io WebSocket (20k trades/s)
├── CandleAggregator.kt            — Agregacja 1-min + 30-min candlesticks
├── DarkPoolFilter.kt              — FILTRUJE (usuwa) dark pool trades
├── AnalysisService.kt             — Volume anomaly detection
└── MongoDB                        — Storage
```

### Paradoks tego projektu

Autor **CELOWO USUWA dark pool trades** ze swojego strumienia danych:

> "The problem with aggregated data is that they contain trades from dark pool stock exchanges, which are used mainly for professional trading by financial institutions. Trade volumes on these exchanges are so large, that no meaningful analysis is possible due to too much noise."

**Oni widzą dark pool jako SZUM do odfiltrowania.** My widzimy go jako SYGNAŁ do wykrycia. Kompletnie przeciwne podejście.

### Co robi LEPIEJ niż my

| Ich przewaga | Szczegóły |
|---|---|
| **Throughput** | 20 000 trades/s na Polygon.io — prawdziwy production-grade |
| **Prawdziwe dane** | Polygon.io WebSocket — nie symulowane, realne US equity trades |
| **Dark pool awareness** | Rozpoznają dark pool exchanges i je filtrują (odwrotność naszego celu) |
| **JVM performance** | Kotlin + Spring WebFlux — reaktywny, nieblokujący I/O |
| **Produkcyjny kod** | Maven, MongoDB, error handling — kod pisany z myślą o produkcji |

### Co robi GORZEJ niż my

| Nasza przewaga | Szczegóły |
|---|---|
| **Cel** | My DETEKUJEMY dark pool, oni go USUWAJĄ |
| **ML** | Brak — same progi wolumenowe |
| **Metody akademickie** | Zero |
| **Web UI** | Brak — tylko backend |
| **Docker** | Brak |
| **Testy** | Brak |
| **Backtest** | Brak |

### Werdykt: **Najlepszy engineering, zły cel (dla nas)**

Najbardziej dopracowany technicznie pod kątem high-frequency data. Ale architektura jest DOKŁADNIE ODWROTNA od naszego celu. To tak jakby mieć najlepszy na świecie system antywłamaniowy... który ignoruje wszystkie włamania i skupia się na pogodzie.

**Co przejąć**: Polygon.io WebSocket jako opcjonalne źródło danych (zamiast symulacji), Spring WebFlux podejście do nieblokującego I/O.

---

## 📊 ZBIORCZE PORÓWNANIE (tylko meritum)

### Gdzie NAPRAWDĘ mają przewagę

| Repo | Co robią lepiej | Czy warto to przejąć? | Priorytet |
|---|---|---|---|
| **#1 Fraud** | TGNN + HAR-BACD-V + XAI (SHAP/LIME) + Bayesian Uncertainty | ✅ TAK — BACD model do TraderType, BNN do uncertainty | 🔴 HIGH |
| **#2 DarkPool** | Kafka + Spark streaming + WebSocket alerts | ✅ TAK — architektura real-time | 🟡 MEDIUM |
| **#3 Stock** | Polygon.io WebSocket (20k trades/s) | ⚠️ Tak, ale jako opcjonalne źródło ($) | 🟢 LOW |
| **#3 Stock** | Świadomość które exchange to dark pool | ✅ TAK — lista venue do filtrowania | 🟢 LOW |

### Gdzie MY mamy przewagę (i to znaczącą)

| Nasz moduł | Oni | Przewaga |
|---|---|---|
| **VPIN (ELO 2011)** | ❌ Nikt nie ma | Jedyna implementacja wśród 4 repo |
| **Iceberg Detection (arXiv:1909)** | ❌ Nikt nie ma | Jedyna implementacja |
| **Backtest (4 strategie)** | ❌ Nikt nie ma | Zero u konkurencji |
| **Testy (157)** | ❌ Nikt nie ma | Zero u konkurencji |
| **CI/CD** | ❌ Nikt nie ma | Tylko my |
| **Public data (FINRA)** | ❌ Nikt nie ma | #3 wymaga płatnego Polygon |
| **Dokumentacja** | ❌ #2 pusta, #1/#3 podstawowa | Jedyny z PLAN.md |
| **Trader Classification** | ❌ Nikt nie ma | Jedyna implementacja GMM |
| **Dark Volume Reconstruction** | ❌ Nikt nie ma | Jedyna implementacja |

### Kluczowy wniosek

NIKT z konkurencji nie łączy **detection + ML + backtest + web + API + Docker + CI/CD + testy** w jednym projekcie. Każdy robi JEDNĄ rzecz dobrze:
- #1 = super ML, zero infra
- #2 = super infra, zerowa detekcja  
- #3 = super engineering, przeciwny cel

**My jesteśmy jedynym PRODUKTEM, nie tylko eksperymentem.**
