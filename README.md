
# 🍔 Food Delivery ETA Prediction System

An end-to-end machine learning project that predicts food delivery time using delivery, traffic, weather, distance, and order-related information.

## 🚀 Project Overview

The system uses an XGBoost regression model to predict the estimated delivery time in minutes.

The project includes:

- Data cleaning
- Exploratory data analysis
- Feature engineering
- Machine learning model comparison
- XGBoost model training
- Model evaluation
- FastAPI REST API
- Streamlit frontend
- Docker containerization

## 🏗️ Architecture

```text
User
 ↓
Streamlit Frontend
 ↓
FastAPI REST API
 ↓
Preprocessing Pipeline
 ↓
XGBoost Model
 ↓
Predicted Delivery Time