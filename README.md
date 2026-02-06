# 🎬 TMDB The Next Blockbuster – End-to-End Machine Learning Project

## Project Overview
This project develops an **end-to-end machine learning pipeline** to support **pre-release decision-making in the movie industry**.  
Using data collected from **The Movie Database (TMDB) official API**, we build models that estimate a movie’s expected success **before release**, based on cast, director, and content attributes.

The project is designed as a **Proof of Value (PoV)** and follows best practices taught in enterprise data science: data extraction, cleaning, feature engineering, modeling, evaluation, and deployment via a simple UI.

---

## Business Problem
Movie studios and streaming platforms must decide **marketing budgets, promotion strategies, and risk exposure** before a movie is released, often with limited information.

**Key questions addressed:**
- Is this movie likely to be a “hit”?
- How much audience interest should we expect?
- How do cast, director, and content choices affect success?

---

## Objectives
We frame the problem using **two complementary ML tasks**:

### 1️⃣ Classification  
Predict whether a movie will be a **“hit”**, defined as being in the **top 20% of popularity**.

### 2️⃣ Regression  
Estimate the movie’s **expected popularity score**, providing a continuous measure of anticipated audience interest.

Both tasks rely **only on pre-release information** to avoid data leakage.

---

## Data Source
- **The Movie Database (TMDB) API**
- Data is collected via authenticated API requests (no web scraping).
- Endpoints used include:
  - `/discover/movie`
  - `/movie/{id}?append_to_response=credits,keywords`
  - `/person/{id}` (for selected actors and directors)

### Dataset Scope
- Movies released between **2018–2023**
- Sampled via popularity-sorted discovery
- ~400 movies (PoC scale)
- Actor and director data enriched with caching to respect rate limits

---

## Feature Engineering

### 🎭 Talent Features
- Individual popularity of **top 5 billed actors**
- Aggregated cast statistics:
  - Average cast popularity
  - Maximum cast popularity
  - Number of “star” actors
- Director popularity

### 🎞 Movie Metadata
- Genres (multi-hot encoded)
- Runtime
- Release month (seasonality)
- Keyword count
- Original language (top-K encoded)

### 🚫 Leakage Control
The following variables are **excluded from model inputs**:
- Popularity (target)
- Vote average / vote count
- Movie title and IDs

They are retained only for evaluation and interpretability.

---

## Modeling Approach

### Models
- **Baseline models** (linear / logistic)
- **Improved models** (tree-based where appropriate)

### Evaluation Metrics
- Classification: ROC-AUC, Precision@Top-K
- Regression: MAE, RMSE, R²
- Business-oriented interpretation of results

---

## User Interface (Streamlit)
A lightweight **Streamlit UI** demonstrates how the model can be used in practice:
- Users input movie characteristics (cast strength, director popularity, genres, runtime)
- Outputs:
  - Probability of being a “hit”
  - Expected popularity score

This illustrates how ML outputs can support **real decision-making**.

---

## Repository Structure

Structure