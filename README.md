# COS30019 Assignment 2B – Traffic-Based Route Guidance System

## Project Overview

This project implements a **Traffic-Based Route Guidance System (TBRGS)** for COS30019 Introduction to AI Assignment 2B.

The system uses historical SCATS traffic flow data from the Boroondara area to train machine learning models that predict traffic flow at SCATS sites. These predictions are then converted into estimated travel times and used to recommend up to five routes with the lowest total travel time.

The project integrates:

- SCATS traffic data preprocessing
- MLP, LSTM, and GRU traffic prediction models
- Model evaluation and comparison
- Travel-time estimation
- SCATS road graph construction
- Assignment 2A CUS1 / Uniform Cost Search integration
- Top-k route recommendation


---

## Project Structure

```text
Assignment2B-Tutor-Style/
│
├── data/
│   ├── data.py
│   ├── processed_scats_15min.csv
│   ├── Scats Data October 2006.csv
│   └── scats_connections.csv
│
├── model/
│   ├── model.py
│   ├── mlp_site_<site>.keras
│   ├── lstm_site_<site>.keras
│   └── gru_site_<site>.keras
│
├── results/
│   ├── batch_training_summary.csv
│   ├── model_comparison_all_sites.csv
│   ├── model_average_comparison.csv
│   ├── best_model_by_site.csv
│   ├── tbrgs_routes.csv
│   └── tbrgs_routes_edge_details.csv
│
├── images/
│   └── prediction graphs
│
├── preprocess_scats.py
├── train.py
├── train_all_sites.py
├── evaluate_all_sites.py
├── main.py
├── traveltime.py
├── search_algorithms.py
├── tbrgs.py
├── requirements.txt
└── README.md
```

---

## Main Files

| File | Purpose |
|---|---|
| `preprocess_scats.py` | Converts the raw SCATS dataset into a cleaned 15-minute time-series dataset |
| `data/data.py` | Loads processed data and creates train/test sequences |
| `model/model.py` | Defines MLP, LSTM, and GRU model architectures |
| `train.py` | Trains one selected model for one selected SCATS site |
| `train_all_sites.py` | Trains MLP, LSTM, and GRU models for all SCATS sites |
| `evaluate_all_sites.py` | Evaluates all trained models and creates comparison CSV files |
| `main.py` | Evaluates one model for one SCATS site and generates a prediction graph |
| `traveltime.py` | Converts predicted traffic flow into estimated travel time |
| `search_algorithms.py` | Contains the adapted Assignment 2A CUS1 / Uniform Cost Search logic |
| `tbrgs.py` | Main Traffic-Based Route Guidance System program |

---

## Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 2. Install Required Packages

```bash
pip install -r requirements.txt
```

Recommended Python version:

```text
Python 3.11
```

TensorFlow may not work correctly with newer Python versions such as Python 3.13 or Python 3.14.

---

## Data Preprocessing

Run:

```bash
python preprocess_scats.py
```

This reads:

```text
data/Scats Data October 2006.csv
```

and creates:

```text
data/processed_scats_15min.csv
```

The processed dataset contains:

```text
SCATS Number
Location
NB_LATITUDE
NB_LONGITUDE
datetime
Flow
```

The raw SCATS dataset contains 96 traffic flow columns per day, representing 15-minute intervals. The preprocessing script converts this into a long-format time-series dataset.

---

## Training Models

### Train One Model for One Site

```bash
python train.py --model mlp --site 2200
python train.py --model lstm --site 2200
python train.py --model gru --site 2200
```

This saves model files such as:

```text
model/mlp_site_2200.keras
model/lstm_site_2200.keras
model/gru_site_2200.keras
```

### Train All Models for All SCATS Sites

```bash
python train_all_sites.py
```

This trains:

```text
40 SCATS sites × 3 models = 120 models
```

The three model types are:

```text
MLP
LSTM
GRU
```

To skip models that already exist:

```bash
python train_all_sites.py --skip-existing
```

To test only selected sites:

```bash
python train_all_sites.py --sites 2200 2825 3001 --epochs 5
```

---

## Model Evaluation

Evaluate all trained models:

```bash
python evaluate_all_sites.py
```

This creates:

```text
results/model_comparison_all_sites.csv
results/model_average_comparison.csv
results/best_model_by_site.csv
```

The evaluation metrics include:

```text
MAE
MSE
RMSE
MAPE
R2 Score
Explained Variance
```

Average model results from testing:

| Model | Sites Evaluated | Average MAE | Average RMSE | Average MAPE | Average R² |
|---|---:|---:|---:|---:|---:|
| GRU | 40 | 28.60 | 40.84 | 72.19% | 0.727 |
| LSTM | 40 | 28.84 | 40.95 | 71.31% | 0.726 |
| MLP | 40 | 28.72 | 42.12 | 68.36% | 0.708 |

Based on RMSE, GRU performed best overall.

---

## SCATS Connection Graph

The TBRGS uses:

```text
data/scats_connections.csv
```

This file defines valid road connections between SCATS sites.

Format:

```csv
from_site,to_site
2000,2200
2200,2000
2200,2820
2820,2200
```

Each row means travel is allowed from `from_site` to `to_site`.

The route guidance system only searches through connections listed in this file.

---

## Travel-Time Estimation

For a road edge:

```text
SCATS A → SCATS B
```

the system:

1. Predicts traffic flow at SCATS site B
2. Converts the predicted 15-minute flow into hourly flow
3. Estimates speed using the traffic flow conversion formula
4. Calculates travel time using distance and speed
5. Adds a 30-second intersection delay
6. Uses this travel time as the edge cost

The distance between two SCATS sites is estimated using their latitude and longitude values.

---

## Running the TBRGS

Run:

```bash
python tbrgs.py --origin 2000 --destination 2820 --datetime "2006-10-15 08:00" --model mlp --k 5
```

Example output:

```text
Traffic-Based Route Guidance System
-----------------------------------
Origin: 2000
Destination: 2820
Time: 2006-10-15 08:00:00
Model: MLP

Routes:
Route 1: 2000 -> 2200 -> 2820 | Travel time: 11.36 minutes
Route 2: 2000 -> 3685 -> 3126 -> 2820 | Travel time: 11.89 minutes
Route 3: 2000 -> 3120 -> 4040 -> 4035 -> 2820 | Travel time: 12.91 minutes
Route 4: 2000 -> 3685 -> 3126 -> 2200 -> 2820 | Travel time: 13.03 minutes
Route 5: 2000 -> 2200 -> 3126 -> 2820 | Travel time: 13.73 minutes
```

The output files are saved to:

```text
results/tbrgs_routes.csv
results/tbrgs_routes_edge_details.csv
```

---

## Assignment 2A and 2B Integration

Assignment 2A involved graph search algorithms. In Assignment 2B, this search logic was adapted into the TBRGS.

The CUS1 / Uniform Cost Search idea was used because the system needs to find routes with the lowest total travel time.

In Assignment 2A:

```text
edge cost = fixed cost from input file
```

In Assignment 2B:

```text
edge cost = predicted travel time
```

The adapted search algorithm is implemented in:

```text
search_algorithms.py
```

and used by:

```text
tbrgs.py
```

---

## Test Cases

The system was tested using 15 test cases covering:

- valid route generation
- MLP, LSTM, and GRU model selection
- morning, midday, and afternoon time periods
- reverse routes
- invalid SCATS site input
- disconnected graph cases
- invalid time intervals
- dates outside the dataset
- insufficient historical data

Main successful route test:

| Route | Path | Travel Time |
|---:|---|---:|
| 1 | 2000 → 2200 → 2820 | 11.36 min |
| 2 | 2000 → 3685 → 3126 → 2820 | 11.89 min |
| 3 | 2000 → 3120 → 4040 → 4035 → 2820 | 12.91 min |
| 4 | 2000 → 3685 → 3126 → 2200 → 2820 | 13.03 min |
| 5 | 2000 → 2200 → 3126 → 2820 | 13.73 min |

---

## Known Limitations

- The SCATS road graph is manually or approximately created.
- The system uses straight-line distance between SCATS coordinates, not exact road distance.
- The dataset only covers October 2006.
- The current version is command-line based and does not include a full GUI.
- TensorFlow may show CPU/GPU warning messages, but these do not stop the program from working.
- Model loading could be optimised by preloading models instead of loading them during route calculation.

---

## References

- COS30019 Assignment 2 specification
- VicRoads / SCATS traffic flow dataset
- TensorFlow / Keras documentation
- Pandas documentation
- NumPy documentation
- Scikit-learn documentation
- Matplotlib documentation
