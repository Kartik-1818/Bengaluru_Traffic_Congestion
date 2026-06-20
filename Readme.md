# 🚦 Bengaluru Event-Driven Traffic Congestion — Response Recommender

> **Gridlock Hackathon 2.0 · Theme 2 · Event-Driven Congestion**  
> An ML-powered dashboard that predicts incident impact and recommends manpower, barricading, and police station deployment in real time.

---

## 📌 Overview

Bengaluru handles thousands of traffic incidents daily — from vehicle breakdowns to VIP movements and protests. This project trains three machine learning models on 8,173 ASTRAM traffic events to:

- Predict **incident priority** (High / Low)
- Predict **road closure requirement** (Yes / No)
- Estimate **incident duration** (hours)
- Recommend **officer count**, **deployment station**, and **barricading** based on the above

The v2 Streamlit app uses **XGBoost** models with rich feature engineering (circular time encoding, haversine hotspot distances, interaction features) achieving **AUC 0.98** and **avg confidence 89%**.

---

## 📁 Project Structure

```
Bengaluru_Traffic_Congestion-main/
│
├── app/
│   └── app_v2.py                        # Streamlit dashboard (v2, XGBoost + feature engineering)
│
├── assets/                              # EDA & model visualisation outputs
│   ├── bengaluru_hotspot_map.html       # Interactive Folium map of congestion hotspots
│   ├── eda_extra1_planned_vs_unplanned.png
│   ├── model1_confusion_matrix.png      # Priority classifier confusion matrix
│   ├── model1_feature_importance.png    # Top-15 feature importances
│   ├── step2_distributions.png          # Feature distributions
│   ├── step3_time_patterns.png          # Hourly / day-of-week traffic patterns
│   ├── step4_severity.png               # Severity index breakdown
│   ├── step5_corridors_zones.png        # Corridor & zone analysis
│   └── step5_zone_time_matrix.png       # Zone × time-block heatmap
│
├── docs/
│   └── vide_lind.md                     # Additional project notes / video link
│
├── models/
│   └── recommendation_engine_bundle_v2.pkl   # Serialised model bundle (XGBoost v2)
│
├── notebook/
│   └── Flipkart_grid_notebook_complete.ipynb # End-to-end EDA + training notebook (Google Colab)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 ML Pipeline

### Dataset
- **Source:** ASTRAM Bengaluru Traffic Events (`Hack_dataset.csv`)
- **Size:** 8,173 rows × 46 columns
- **Key columns:** `event_type`, `event_cause`, `latitude`, `longitude`, `priority`, `requires_road_closure`, `start_datetime`, `closed_datetime`, `corridor`, `zone`, `police_station`

### Preprocessing (Notebook)
1. Drop fully-null columns (`map_file`, `comment`, `meta_data`)
2. Parse all datetime columns; remove rows with `end_datetime < start_datetime`
3. Drop columns with >90% missing values
4. Replace placeholder `0.0` in `endlatitude` / `endlongitude` with `NaN`
5. Derive `duration_hrs` from `(closed_datetime − start_datetime)`
6. Extract time features: `hour`, `day_of_week`, `month_num`
7. Engineer binary flags: `is_weekend`, `is_peak_hour` (7–9 AM, 5–8 PM), `is_night` (10 PM–6 AM)
8. Normalise `event_cause` (lowercase + strip)

### Feature Engineering (v2 App)
- **Circular time encoding:** `sin/cos` of hour, month, day-of-week
- **Haversine distances** to 6 known congestion hotspots (MG Road, Silk Board, Hebbal, Marathahalli, Whitefield, Electronic City)
- **Interaction features:** `peak_x_cause`, `weekend_x_cause`
- **Cause severity score** from lookup map
---

### Exploratory Data Analysis

**Event Distributions** — cause breakdown, planned vs unplanned, priority, status
![Event distributions](assets/step2_distributions.png)

**Time Patterns** — incidents by hour/day/month, hour×day heatmap
![Time patterns](assets/step3_time_patterns.png)

**Severity Analysis** — duration by cause, closure rate by cause, severity index distribution
![Severity analysis](assets/step4_severity.png)

**Corridor & Zone Analysis** — top corridors by risk, zone-wise incident load, police station load
![Corridor and zone analysis](assets/step5_corridors_zones.png)

**Zone × Time Manpower Matrix** — deployment planning heatmap
![Zone time matrix](assets/step5_zone_time_matrix.png)

**Planned vs Unplanned — Cause Comparison**
![Planned vs unplanned](assets/eda_extra1_planned_vs_unplanned.png)

**Priority Classifier — Confusion Matrix**
![Priority confusion matrix](assets/model1_confusion_matrix.png)

**Priority Classifier — Feature Importance**
![Priority feature importance](assets/model1_feature_importance.png)

**Interactive Hotspot Map**
An interactive Folium heatmap of all incidents, with high-severity and road-closure events
marked individually, is available at [`assets/bengaluru_hotspot_map.html`](assets/bengaluru_hotspot_map.html).
GitHub doesn't render embedded HTML inline, so clone the repo and open the file locally
(or open it directly from the file browser above) to interact with it.

---

### Models

| # | Task | Algorithm | Key Metric |
|---|------|-----------|------------|
| 1 | Priority classification (High / Low) | XGBoost (v2) / RandomForest (v1) | AUC 0.98 · F1 0.90 |
| 2 | Road closure classification (Yes / No) | XGBoost (`scale_pos_weight` for imbalance) | Threshold-optimised |
| 3 | Duration regression (hours) | XGBoost Regressor (log-transformed target) | MAE in hours |

**Imbalance handling:** `scale_pos_weight = neg/pos` passed to XGBoost closure model.  
**Leakage prevention:** `priority_score` and `closure_score` excluded from classifier features; `severity_index` used only for duration regression.

### Model Bundle (`recommendation_engine_bundle_v2.pkl`)
Serialised with `joblib`, the bundle contains:
- `priority_model`, `closure_model`, `duration_model`
- `priority_feature_cols`, `closure_feature_cols`, `duration_feature_cols`
- `cat_features`, `num_features`, `cat_features_fixed`, `num_features_fixed`
- `closure_threshold` (optimised for class imbalance)
- `cause_score_map`, `manpower_map`
- `zone_station_map`, `station_coords`, `corridor_coords`
- `hotspots` (lat/lon of 6 key congestion points)

---

## 🖥️ Streamlit App (`app/app_v2.py`)

### What It Does
1. Accepts an incoming traffic event (type, cause, GPS, time, zone)
2. Auto-detects the nearest **corridor** and **hotspot** via haversine distance
3. Runs all three models and computes a **severity score** (0–11)
4. Outputs:
   - Risk level (Low / Medium / High / Critical)
   - Predicted priority + confidence gauge
   - Road closure prediction + confidence gauge
   - Estimated duration
   - Recommended officer count
   - Recommended police station (zone lookup or GPS nearest-neighbour)
   - Barricading recommendation
   - Action checklist
   - Live map with hotspot overlays

### Quick Presets
Three one-click presets for demo purposes:
- Morning Accident — MG Road, 8 AM
- VIP Movement — City Centre, 10 AM
- Night Protest — Silk Board, 11 PM

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/Kartik-1818/Bengaluru_Traffic_Congestion
cd Bengaluru_Traffic_Congestion
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app/app_v2.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** The pre-trained model bundle (`models/recommendation_engine_bundle_v2.pkl`) is included. No retraining is required to run the app.

---

## 📓 Reproducing the Training (Notebook)

The full pipeline lives in `notebook/Flipkart_grid_notebook_complete.ipynb` and was developed on **Google Colab**.

### Steps to re-run
1. Upload `Hack_dataset.csv` to your Google Drive at `MyDrive/Hack_dataset.csv`
2. Open the notebook in Colab and mount Drive
3. Run all cells top-to-bottom:
   - **Cells 1–N:** Data loading, cleaning, EDA, chart exports
   - **ML Section:** Feature engineering → Model 1 (Priority) → Model 2 (Closure) → Model 3 (Duration)
   - **Bundle Section:** Saves `recommendation_engine_bundle_v2.pkl` to Drive
4. Copy the bundle into `models/` locally before running the app

### Install notebook-only dependencies
```bash
pip install xgboost imbalanced-learn plotly folium
```

---

## 📦 Dependencies (`requirements.txt`)

```
pandas
numpy
scikit-learn>=1.2
xgboost>=1.7
streamlit
joblib
folium
seaborn
matplotlib
```

---

## 📊 EDA Highlights

| Insight | Detail |
|---------|--------|
| Dataset size | 8,173 traffic events, 46 raw columns |
| Event split | ~majority unplanned; minority planned |
| Peak hours | 7–9 AM and 5–8 PM |
| Top hotspots | Silk Board, Marathahalli, Hebbal, MG Road |
| Severity range | 0–11 composite score (cause + type + priority + closure) |

Visual outputs are saved to `assets/` and include distribution plots, time-pattern charts, corridor/zone heatmaps, and an interactive Folium map.


## 🗺️ Known Hotspot Coordinates

| Hotspot | Latitude | Longitude |
|---------|----------|-----------|
| MG Road | 12.9766 | 77.6075 |
| Silk Board | 12.9174 | 77.6228 |
| Hebbal | 13.0358 | 77.5970 |
| Marathahalli | 12.9563 | 77.7010 |
| Whitefield | 12.9698 | 77.7500 |
| Electronic City | 12.8399 | 77.6770 |

---

## 📄 License

This project was developed as a hackathon submission. Dataset rights belong to ASTRAM / the hackathon organisers.
