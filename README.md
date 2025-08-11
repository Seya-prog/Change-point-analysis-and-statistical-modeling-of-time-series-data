# Change-Point Analysis and Statistical Modeling of Time Series Data

This project focuses on detecting change points in time series data and applying various statistical models to analyze temporal patterns.

## Project Structure

```
├── data/                   # Raw and processed datasets
│   ├── raw/               # Original data files
│   ├── processed/         # Cleaned and preprocessed data
│   └── sample/            # Sample datasets for testing
├── src/                   # Source code
│   ├── data_processing/   # Data loading and preprocessing
│   ├── change_point/      # Change point detection algorithms
│   ├── models/            # Statistical models
│   ├── visualization/     # Plotting and visualization utilities
│   └── utils/             # Helper functions
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── results/               # Output results and figures
├── config/                # Configuration files
└── docs/                  # Documentation

## Installation

1. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Task 3: Interactive Dashboard (Docker Setup)

The project includes a complete Flask + React dashboard that can be run using Docker:

#### Quick Start with Docker
```bash
# Windows
docker-start.bat

# Linux/Mac
chmod +x docker-start.sh
./docker-start.sh
```

#### Manual Docker Setup
```bash
# Build and start all services
docker-compose up --build -d

# Access the dashboard
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000/api
```

#### Local Development
```bash
# Backend (Flask)
cd backend
python app.py

# Frontend (React)
cd frontend
npm install
npm start
```

### Previous Tasks

#### Task 1: Data Processing and Analysis
```bash
python src/data_processing/main.py
```

#### Task 2: Bayesian Change-Point Analysis
```bash
python bayesian_changepoint_analysis.py
```

## Features

- Multiple change-point detection algorithms
- Statistical modeling techniques
- Time series visualization
- Model evaluation and comparison
- Automated reporting

## Contributing

[Guidelines to be added]
