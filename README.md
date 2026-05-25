# Assignment 2B – Traffic-Based Route Guidance System (TBRGS)

## Overview

This project implements a Traffic-Based Route Guidance System (TBRGS) for the Boroondara road network using machine learning and graph-based routing techniques.

The system predicts future traffic flow using deep learning models and converts predicted traffic conditions into estimated travel times for route optimisation.

The project follows a modular tutor-style structure inspired by the provided traffic flow prediction framework.

---

# Project Objectives

The project consists of four major tasks:

1. Data preprocessing and traffic dataset preparation
2. Machine learning-based traffic flow prediction
3. Travel time estimation using predicted traffic flow
4. Integration with graph-based route guidance algorithms

---

# Dataset

Dataset source:
- SCATS Traffic Volume Data
- Boroondara Council
- October 2006

Dataset characteristics:
- 40 SCATS sites
- 15-minute traffic intervals
- 96 traffic readings per day (`V00 – V95`)
- Traffic flow counts for multiple intersections

---

# Project Structure

```text
ASSIGNMENT2B-TUTOR-STYLE/

├── data/
│   ├── data.py
│   ├── processed_scats_15min.csv
│   └── Scats Data October 2006.csv
│
├── images/
│   └── mlp_site_2200.png
│
├── model/
│   ├── model.py
│   └── mlp_site_2200.keras
│
├── results/
│   ├── mlp_site_2200_loss.csv
│   └── mlp_site_2200_metrics.txt
│
├── venv/
│
├── main.py
├── preprocess_scats.py
├── README.md
├── requirements.txt
├── train.py
├── travel_time.py
└── .gitignore
```

---

# Implemented Models

The system currently supports:

- MLP (Multi-Layer Perceptron)
- LSTM (Long Short-Term Memory)
- GRU (Gated Recurrent Unit)

---

# Data Processing

The raw SCATS dataset is converted into a time-series format suitable for machine learning.

Processing includes:
- removing empty columns
- reshaping traffic intervals (`V00 – V95`)
- creating datetime values
- sorting by SCATS site and timestamp
- converting traffic flow values to numerical format

The final processed dataset is saved as:

```text
data/processed_scats_15min.csv
```

---

# Machine Learning Pipeline

## Training

Models are trained using:

```powershell
python train.py --model mlp --site 2200
```

Available models:

- mlp
- lstm
- gru

Example:

```powershell
python train.py --model lstm --site 2200
```

---

# Evaluation

Evaluate trained models using:

```powershell
python main.py --model mlp --site 2200
```

Evaluation metrics include:
- MAE
- MSE
- RMSE
- MAPE
- R² Score
- Explained Variance Score

Prediction graphs are automatically saved to:

```text
images/
```

Evaluation metrics are saved to:

```text
results/
```

---

# Travel Time Estimation

The `travel_time.py` module converts predicted traffic flow into estimated vehicle speed and travel time.

The implementation includes:
- flow-to-speed conversion
- distance calculation using latitude/longitude
- travel time estimation
- intersection delay estimation

---

# Requirements

Install dependencies using:

```powershell
pip install -r requirements.txt
```

Required libraries:

```text
pandas
numpy
matplotlib
scikit-learn
tensorflow
pydot
graphviz
```

---

# Virtual Environment Setup

Create virtual environment:

```powershell
py -3.13 -m venv venv
```

Activate virtual environment:

```powershell
.\venv\Scripts\activate
```

---

# Example Workflow

## Step 1 — Process dataset

```powershell
python preprocess_scats.py
```

## Step 2 — Train model

```powershell
python train.py --model mlp --site 2200
```

## Step 3 — Evaluate model

```powershell
python main.py --model mlp --site 2200
```

## Step 4 — Estimate travel time

```powershell
python travel_time.py
```

---

# Current Results (MLP Example)

| Metric | Result |
|---|---|
| MAE | 16.81 |
| RMSE | 24.90 |
| R² Score | 0.84 |

The MLP model successfully captures general traffic flow behaviour while showing larger prediction errors during sudden congestion spikes.

---

# Future Improvements

Potential future improvements include:
- integration with Assignment 2A routing algorithms
- top-k shortest path generation
- dynamic route optimisation
- improved congestion modelling
- additional deep learning architectures
- real-time traffic integration

---

# Authors

Assignment 2B – Introduction to AI  
Swinburne University of Technology  
Semester 1 – 2026