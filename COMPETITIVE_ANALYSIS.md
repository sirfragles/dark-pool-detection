# 🕵️ Dark Pool Detection — Analiza istniejących repozytoriów

Analiza 4 repozytoriów GitHub związanych z wykrywaniem dark pool activity.
Data badania: 2026-05-13

---

## 1. pranay0703/dark-pool-fraud-detection

🔗 https://github.com/pranay0703/dark-pool-fraud-detection

### Metryki
| Pole | Wartość |
|---|---|
| Język | Python |
| Gwiazdki | 0 |
| Ostatnia aktualizacja | Sep 19, 2025 |
| Licencja | Brak |

### Opis
System AI do wykrywania fraudu w dark pool tradingu w czasie rzeczywistym, wykorzystujący **Temporal Graph Neural Networks (TGNN)** i architektury **Transformer**. To najbardziej zaawansowany ML spośród analizowanych repo.

### Architektura kodu
```
dark_pool_fraud_detection/
├── configs/              # Konfiguracje modeli
├── src/
│   ├── data_pipeline/    # Pipeline pobierania i przetwarzania danych
│   ├── models/           # Definicje modeli (TGNN, Transformer)
│   ├── training/         # Logika trenowania
│   └── inference/        # Inferencja w czasie rzeczywistym
├── setup.py              # 10.7 KB — rozbudowana instalacja
├── train.py              # 4.6 KB — skrypt treningowy
├── demo.py               # 20.5 KB — demo aplikacja
├── evaluate.py           # 18 KB — ewaluacja modeli
└── requirements.txt      # 312 B — zależności
```

### Stack technologiczny
- **Modele**: Temporal GNN + Transformer (PyTorch)
- **Dane**: Symulowane dark pool (prawdopodobnie syntetyczne)
- **Pipeline**: data_pipeline → training → inference
- **Wielkość**: ~54 KB kodu Python (src + skrypty)

### Ciekawe aspekty
1. **Temporal GNN**: Grafowe sieci neuronowe z wymiarem czasowym — idealne do modelowania relacji między traderami, venue'ami i transakcjami w czasie
2. **Transformer**: Wykrywanie sekwencyjnych wzorców fraudu w strumieniu transakcji
3. **setup.py (10.7 KB)**: Sugeruje rozbudowany system z wieloma zależnościami i konfiguracjami
4. **demo.py (20.5 KB)**: Największy plik — prawdopodobnie zawiera wizualizacje i interfejs demo
5. **evaluate.py (18 KB)**: Rozbudowana ewaluacja modeli

### Podobieństwa do naszego projektu
- Python-first
- Symulowane dane
- Modularna architektura (data/model/training)
- Podobny pipeline: dane → detekcja → predykcja

### Różnice
- Frauds ≠ dark pool detection — inne zadanie (klasyfikacja fraudu vs detekcja aktywności)
- TGNN + Transformer vs nasz XGBoost + LSTM
- Brak web UI / API
- Brak Docker / CI/CD
- Brak testów

---

## 2. sagarvrma/darkpooldetector

🔗 https://github.com/sagarvrma/darkpooldetector

### Metryki
| Pole | Wartość |
|---|---|
| Język | JavaScript / Python / Scala |
| Gwiazdki | 0 |
| Ostatnia aktualizacja | Mar 8, 2026 |
| Licencja | Brak |

### Opis
Platforma czasu rzeczywistego do wykrywania dark pool i block trade activity. Zbudowana na **Apache Kafka** (event streaming), **Spark Structured Streaming** (anomaly detection), i **React** (dashboard). Wszystko skonteneryzowane Dockerem.

### Architektura
```
darkpooldetector/
├── data-ingestion/       # Kafka producers — pobieranie danych rynkowych
├── spark-jobs/           # Spark Structured Streaming — detekcja anomalii
├── api/                  # Backend API (Node.js/Express?)
├── dashboard/            # React frontend
└── docker-compose.yml    # Pełna orkiestracja (1.9 KB)
```

### Stack technologiczny
- **Streaming**: Apache Kafka (event-driven)
- **Analityka**: Spark Structured Streaming (big data processing)
- **Frontend**: React (live dashboard)
- **Backend**: Node.js API
- **Infrastruktura**: Docker Compose (wieloserwisowy)
- **Języki**: JS + Python + Scala = 3 języki w jednym projekcie

### Ciekawe aspekty
1. **Kafka + Spark**: Enterprise-grade stack do real-time stream processingu — skaluje się do milionów eventów/s
2. **Docker Compose (1.9 KB)**: Sugeruje złożoną infrastrukturę wieloserwisową
3. **React Dashboard**: Live wizualizacja dark pool activity
4. **Block trade detection**: Oprócz dark pool — wykrywanie dużych transakcji blokowych
5. **3 języki programowania**: JS (frontend + API), Python (Spark jobs), Scala (Spark jobs)

### Podobieństwa do naszego projektu
- Docker Compose
- Web dashboard (React vs nasz Flask/Streamlit)
- Anomaly detection (Spark vs nasz Python)
- API layer

### Różnice
- Enterprise stack (Kafka/Spark) vs nasz lekki Python pipeline
- 3 języki vs nasz pure Python
- Block trade focus (nie tylko dark pool)
- Brak ML (same reguły/anomalie?)
- Brak testów, brak CI/CD
- Pusty README — projekt we wczesnej fazie

---

## 3. stefluhh/realtime-stock-exchange-analysis

🔗 https://github.com/stefluhh/realtime-stock-exchange-analysis

### Metryki
| Pole | Wartość |
|---|---|
| Język | Kotlin |
| Gwiazdki | 0 |
| Ostatnia aktualizacja | Feb 5, 2025 |
| Licencja | Brak |

### Opis
Aplikacja Spring Boot do analizy algorytmicznej US stock trades w czasie rzeczywistym, używająca **Polygon.io WebSocket API**. Przetwarza do **20 000 transakcji na sekundę**, filtruje dark pool data i aplikuje niestandardowe strategie analityczne. Zbudowana w **Kotlin + MongoDB**.

### Architektura
```
realtime-stock-exchange-analysis/
├── src/main/kotlin/com/stefluhh/
│   ├── StockpriceStreamingAdapter.kt  # Polygon.io WebSocket stream
│   ├── AnalysisService.kt             # Strategie analityczne
│   ├── CandleAggregator.kt            # Agregacja 1-min/30-min candlesticks
│   └── DarkPoolFilter.kt              # Filtrowanie dark pool trades
├── pom.xml                 # 6.3 KB — Maven dependencies
└── readme.md               # 3 KB — dokumentacja
```

### Stack technologiczny
- **Język**: Kotlin (JVM)
- **Framework**: Spring Boot (WebFlux — reaktywny)
- **Baza danych**: MongoDB
- **Dane**: Polygon.io WebSocket API (płatne, real-time)
- **Build**: Maven

### Kluczowe koncepty techniczne
1. **Polygon.io Trades WebSocket**: Streamuje KAŻDĄ pojedynczą transakcję z US exchanges (do 20k/s)
2. **Dark Pool Filter**: Filtruje transakcje z dark pool exchanges ponieważ "trade volumes on these exchanges are so large, that no meaningful analysis is possible due to too much noise" — **ODWROTNY problem niż nasz** — oni usuwają dark pool, my go wykrywamy
3. **Agregacja candlestick**: Z pojedynczych trade'ów tworzy 1-minutowe i 30-minutowe świece
4. **Volume anomaly detection**: Wykrywanie skoków wolumenu 100-1000% w ciągu 1-2 minut
5. **Wyzwanie**: Volume naturalnie spike'uje na zamknięciu sesji → filtrowanie ostatnich 40 minut

### Podobieństwa do naszego projektu
- Real-time processing
- Volume anomaly detection
- Dark pool awareness (ale odwrotne podejście)
- Modularna architektura

### Różnice
- **Kotlin/JVM vs Python** — kompletnie inny ekosystem
- **Usuwanie dark pool** vs wykrywanie go
- Polygon.io (płatne API) vs nasze publiczne dane
- Brak ML (reguły vs modele)
- Brak Docker, testów, CI/CD
- Production-grade performance (20k trades/s)

---

## 4. skyreapermodder/Dark-Pool-Whale-Order-Flow-Sniffer

🔗 https://github.com/skyreapermodder/Dark-Pool-Whale-Order-Flow-Sniffer

### Metryki
| Pole | Wartość |
|---|---|
| Język | Binary (Windows .exe) |
| Gwiazdki | 0 |
| Ostatnia aktualizacja | 3 godziny temu |
| Licencja | Brak |

### Opis
Narzędzie desktopowe (Windows) do wykrywania dużych ukrytych zleceń ("whale orders") i analizy dark pool activity na rynkach **kryptowalut**. Śledzi order flow, dane blockchain, volume spikes i liquidity w czasie zbliżonym do rzeczywistego.

### Architektura
```
Dark-Pool-Whale-Order-Flow-Sniffer/
├── README.md                           # 5.7 KB — instrukcja
└── unprovokable/
    └── Pool-Sniffer-Flow-Whale-Dark-Order-v2.1-beta.4.zip  # Binarka
```

### Cechy (z README)
- 🐳 **Whale detection**: Wykrywanie dużych zleceń od instytucji
- 🌑 **Dark pool activity**: Sygnały z off-exchange trades
- 📊 **Volume spike alerts**: Powiadomienia o nietypowym wolumenie
- 🔗 **Blockchain monitoring**: Śledzenie transakcji on-chain (Ethereum)
- 📈 **Wykresy i alerty**: Wizualna prezentacja sygnałów
- 🪙 **Multi-asset**: Ethereum + inne główne kryptowaluty
- 🖥️ **GUI**: Interfejs desktopowy (bez programowania)

### Stack technologiczny (nieznany — zamknięte źródło)
- **Platforma**: Windows only (.exe installer)
- **Dane**: Prawdopodobnie API giełd krypto (Binance, Coinbase?)
- **Blockchain**: Ethereum RPC / WebSocket
- **UI**: Desktop GUI (Electron? C#? Python + Qt?)

### Ciekawe aspekty
1. **Krypto ≠ TradFi**: Jedyny projekt skupiony na krypto, nie akcjach
2. **Zamknięte źródło**: Brak kodu — tylko binarka .exe
3. **v2.1-beta.4**: Aktywnie rozwijany (update 3h temu)
4. **"Unprovokable"**: Nietypowa nazwa katalogu
5. **Desktop-only**: Brak web UI, API, Dockera
6. **Łatwość użycia**: "You do not need any programming skill" — target: retail trader

### Podobieństwa do naszego projektu
- Whale/large order detection (nasz iceberg detection)
- Dark pool activity monitoring
- Volume spike alerts (nasz anomaly detection)
- Real-time/near real-time

### Różnice
- **Zamknięte źródło** — nie możemy analizować kodu
- **Krypto vs Equities** — inny rynek, inne dane
- **Desktop GUI vs Web** — inny model dystrybucji
- **Brak ML** (prawdopodobnie reguły/heurystyki)
- **Windows-only** — brak cross-platform
- **Brak API / testów / CI/CD**
- **Brak Docker** — manualna instalacja

---

## 📊 Porównanie z naszym projektem (sirfragles/dark-pool-detection)

| Kryterium | Nasz | #1 Fraud Detection | #2 DarkPool Detector | #3 Stock Analysis | #4 Whale Sniffer |
|---|---|---|---|---|---|
| **Język** | Python | Python | JS+Py+Scala | Kotlin | Binary .exe |
| **Open source** | ✅ Tak | ✅ Tak | ✅ Tak | ✅ Tak | ❌ Nie |
| **Docker** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **CI/CD** | ✅ GitHub Actions | ❌ | ❌ | ❌ | ❌ |
| **Testy** | ✅ 157 testów | ❌ | ❌ | ❌ | ❌ |
| **Web UI** | ✅ Flask + Streamlit | ❌ (CLI) | ✅ React | ❌ | ✅ Desktop GUI |
| **REST API** | ✅ 7 endpointów | ❌ | ✅ | ❌ | ❌ |
| **ML/Deep Learning** | ✅ XGBoost + LSTM + Clustering | ✅ TGNN + Transformer | ❌ (reguły) | ❌ (reguły) | ❌ (reguły?) |
| **Real-time** | ⚠️ Symulowane | ✅ | ✅ Kafka/Spark | ✅ 20k trades/s | ✅ Near real-time |
| **Public data** | ✅ FINRA + YFinance | ❌ | ❌ | ❌ Polygon ($) | ❌ Krypto API |
| **Backtest** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Alert system** | ✅ JSONL/CSV | ❌ | ⚠️ Dashboard | ❌ | ✅ Desktop alerts |
| **VPIN** | ✅ ELO (2011) | ❌ | ❌ | ❌ | ❌ |
| **Iceberg detection** | ✅ arXiv:1909.09495 | ❌ | ❌ | ❌ | ⚠️ Whale orders |
| **Trader classification** | ✅ GMM/KMeans | ❌ | ❌ | ❌ | ❌ |
| **Dokumentacja** | ✅ README 17 KB | ✅ README 4.6 KB | ❌ Pusta | ✅ readme 3 KB | ✅ README 5.7 KB |
| **Licencja** | ❌ Brak | ❌ Brak | ❌ Brak | ❌ Brak | ❌ Brak |
| **Rynek** | US Equities | US Equities | US Equities | US Equities | Crypto |
| **Aktywność** | Aktywny (dziś) | Sep 2025 | Mar 2026 | Feb 2025 | Aktywny (3h temu) |

## 🎯 Wnioski

### Co nas wyróżnia

1. **Najbardziej kompletny projekt**: Jesteśmy jedynym repo które ma testy + CI/CD + Docker + Web UI + API + ML + backtest + alerty w jednym

2. **Jedyny z VPIN**: Implementacja Volume-Synchronized PIN (Easley, López de Prado, O'Hara 2011) — nikt inny tego nie ma

3. **Jedyny z iceberg detection**: Metoda z literatury akademickiej (arXiv:1909.09495)

4. **Jedyny z backtestem**: Możliwość testowania strategii tradingowych na sygnałach

5. **Public data first**: Nie wymagamy płatnych API (Polygon) — wszystko z publicznych źródeł

### Co możemy poprawić (inspiracje z innych repo)

| Od kogo | Co przejąć |
|---|---|
| #2 DarkPool Detector | **Real-time streaming** (Kafka) — nasz pipeline jest batchowy. Warto dodać WebSocket ingest. |
| #3 Stock Analysis | **High-throughput processing** (20k trades/s) — nasz symulator to tylko prototyp. Polygon.io API jako opcjonalny source. |
| #3 Stock Analysis | **Dark pool filter** (odwrotne podejście) — ciekawe, że niektórzy UWAŻAJĄ dark pool za szum do odfiltrowania |
| #1 Fraud Detection | **TGNN + Transformer** — bardziej zaawansowane ML niż nasz XGBoost/LSTM. Warto jako Faza 9. |
| #4 Whale Sniffer | **Krypto support** — rozszerzenie na rynek krypto (Ethereum mempool analysis) |

### Czego unikać

- ❌ **Zamknięte źródło** (#4) — brak transparentności
- ❌ **3 języki** (#2) — koszmar utrzymania
- ❌ **Płatne API jako jedyne źródło** (#3) — vendor lock-in
- ❌ **Brak testów** (wszyscy poza nami) — nieprofesjonalne
