# DS4420 Project Phase I: SVD-Based Matrix Factorization

## Overview
This project is a proof-of-concept implementation of a recommendation system using **SVD-based matrix factorization** on the MovieLens 1M dataset. The goal is to predict user ratings for movies based on historical user-item interactions.

For Phase I, this repository focuses on the **first non-Bayesian model**: a manual matrix factorization baseline implemented in **Python + NumPy**.

---

## Dataset
We use the MovieLens 1M-style data files:

- `ratings.dat`
- `movies.dat`
- `users.dat`

The ratings file contains entries in the format:

UserID::MovieID::Rating::Timestamp

In our current run:

- **Total ratings:** 1,000,209
- **Total users:** 6,040
- **Total items:** 3,706

---

## Model
We implement a standard latent-factor collaborative filtering model with user and item biases:

\[
\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i
\]

where:

- \(\mu\) = global mean rating  
- \(b_u\) = user bias  
- \(b_i\) = item bias  
- \(p_u\) = latent vector for user \(u\)  
- \(q_i\) = latent vector for item \(i\)  

The model is trained using **stochastic gradient descent (SGD)** with **L2 regularization**.

---
