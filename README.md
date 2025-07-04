# Elo-Serve-Tennis

An interactive tennis match prediction app using Elo ratings and machine learning. Calculates win probabilities and betting expected value from ATP data (2020–2025) with a Streamlit web interface.

## Overview

Elo Serve Tennis Predictor calculates real-time win probabilities for ATP tennis matches by combining:

- Historical match data (ATP 2020–2025)  
- Elo rating calculations updated from match results  
- A Random Forest model trained on Elo differences to estimate winning probabilities  
- Expected Value (EV) calculations using bookmaker odds for betting insights  

The app provides a clean, user-friendly interface where you input two player names and bookmaker odds, then get a detailed prediction with win chances and betting recommendations.

## Features

- Dynamic Elo rating calculation from raw ATP match data  
- Machine learning-based win probability prediction  
- Expected Value (EV) computation for smart betting decisions  
- Streamlit web app with live user inputs and results display  
- Easily extendable for additional betting markets or data sources  

## Project Structure

├── eloserveapp.py                  # Main Streamlit UI app  
├── EloServeTennisPublicV1.py       # Elo rating and prediction functions  
├── ATP20-25AllData.xlsx            # Raw ATP match data  
├── requirements.txt                # Python dependencies  
├── README.md                      # Project overview and instructions  

## Getting Started

### Prerequisites

- Python 3.8+  
- pip  

### Installation

1. Clone the repo:

    git clone https://github.com/genenikolaev/Elo-Serve-Tennis.git  
    cd elo-serve-tennis

2. Install all required dependencies:

    pip install -r requirements.txt

### Running the App

Run this command to start the Streamlit app:

    streamlit run eloserveapp.py

Then open the URL that appears in your browser.
