# Netflix-Style Movie Recommendation System

## Project Overview
This project builds a Netflix-style movie recommendation system to predict how users would rate movies they have not seen.  
We train and compare several machine learning models using historical movie rating data.

## Team Members
- Yujie Chen
- Xinwen Zhang
- Zehao Shi

## Dataset
We use the **MovieLens movie rating dataset**, a standard benchmark dataset for recommender system research.

The dataset contains:
- user IDs
- movie IDs
- user ratings

This allows us to build a **user–movie interaction matrix** for recommendation modeling.

## Models
We implement and compare three recommendation models:

1. **Matrix Factorization (SVD)**  
2. **Neural Collaborative Filtering (Neural CF)**  
3. **Bayesian Probabilistic Matrix Factorization (Bayesian PMF)**

## Repository Structure
data/ dataset files
notebooks/ exploratory analysis and experiments
src/ model implementations
README.md project description

## Phase 1
For Phase 1, we build a **baseline recommendation model** and perform initial experiments on the MovieLens dataset.
