# AB Data Challenge — Project Management Course UPF

## Modeling the impact of weather on water consumption in Barcelona

--- 

### Team

- Luca Franceschi
- Carolina Carbone
- Jan Corcho
- Maggie O'Reilly
- Júlia Othats-Dalès

---

### About this Repository

This repository was developed in the context of the Aigües de Barcelona (AB) Data Challenge, adapted for our Project Management course.
Our team was not formally selected for the challenge; instead, we use the challenge’s theme and structure as a case study to apply project management principles and practice data analysis workflows.

All data used here are synthetic or publicly available sample datasets designed to mimic the format and complexity of the official challenge data.
The purpose is to demonstrate project planning, data preprocessing, exploratory analysis, and visualization, not to produce competition-grade predictive results.

> Disclaimer: This project is an academic exercise based on sample data and is not an official submission to the Aigües de Barcelona Data Challenge.

--- 

### Objectives

- Explore water consumption patterns across districts, uses, and time in the city of Barcelona.
- Apply project management methodologies (WBS, risk analysis, progress tracking) to structure the data project.
- Demonstrate an analytical workflow from data collection to visualization and interpretation.

--- 

### Data Description

The sample dataset follows the same structure as the AB Challenge data:

| Column | Description |
| ----------- | ----------- |
| CensusSection	| Census section identifier | 
| District	| Numeric district code (1–10) |
| District_Name	| Name of the district (mapped) |
| Municipality	| Municipality (Barcelona) |
| Date	| Date of observation (YYYY-MM-DD) |
| Use	| Type of water use (Domestic, Commercial, Industrial) |
| NumMeters	| Number of active meters in that section |
| Consumption_L_day	| Total daily water consumption in liters |

## Python environment setup

Tested with Python version 3.13.7

```
python3.13 -m venv .venv
```

```
source .venv/bin/activate
```

```
pip install -r requirements.txt
```

> ! Remember to drop the datasets in the /data directory and **not upload sensible information**!
