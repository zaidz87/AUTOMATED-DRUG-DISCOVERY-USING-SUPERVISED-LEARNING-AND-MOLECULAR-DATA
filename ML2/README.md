# Automated Drug Discovery Platform

A machine learning-based platform for automated drug discovery using molecular properties to predict compound activity.

## Features

- Data preprocessing and exploratory data analysis
- Machine learning models (Random Forest and Logistic Regression)
- Interactive visualizations using Plotly
- Modern, user-friendly Streamlit interface
- Real-time predictions for new compounds

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your dataset (CSV format) with the following columns:
- SMILES: Molecular structure
- MolWt: Molecular weight
- LogP: Lipophilicity
- NumHDonors: Number of hydrogen bond donors
- NumHAcceptors: Number of hydrogen bond acceptors
- TPSA: Topological polar surface area
- Activity: Binary classification (1 = active, 0 = inactive)

3. Explore the data, train models, and make predictions through the interactive interface.

## Project Structure

- `app.py`: Main Streamlit application
- `model.py`: Machine learning model implementations
- `utils.py`: Helper functions for data processing and visualization
- `requirements.txt`: Project dependencies
