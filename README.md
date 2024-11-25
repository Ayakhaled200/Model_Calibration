# Knowledge Base Dataset Model Evaluation

This repository contains the evaluation of **9 deployed machine learning models** on a **knowledge base dataset**, utilizing its **meta-features** for deeper insights. The project aims to analyze the performance of these models using various metrics and calibration techniques to ensure reliability.

---

## Project Overview

- **Dataset**:
  The dataset is a collection of structured knowledge-based records, enhanced with **meta-features** that describe the characteristics of the data. These meta-features help in understanding model behavior and improving evaluation quality.

- **Model Evaluation**:
  We tested **9 APIs**, each corresponding to a deployed model, to evaluate their predictive performance and reliability.

- **Metrics Used**:
  The following metrics were used to assess model performance:
  - **Balanced Accuracy**
  - **Recall**
  - **F1 Score**

  - **Calibration**:
  Calibration methods like **ECE** and **Reliability Diagram** were applied to ensure the predicted probabilities were aligned with actual outcomes, enhancing model interpretability and usability.

---

## Repository Structure

- **`/data`**:
  Contains the dataset used for evaluation.

- **`/Testing_Scripts`**:
  Jupyter notebooks and python scripts for Api's model evaluation and calibration.

- **`/predictions`**:
  > **Note**: This folder contains predictions from the tested models. It is required only for running the **evaluation notebook**. On its own, it holds no significant value.

- **`Calibration report.`**:
  Stores results, metrics, and analysis of the results like measuring calibration, overfitting and underfitting.

---

## Authors

- **Aya Khaled**
- **Aisha Hagar**

This project was a collaborative effort, focusing on model evaluation, calibration, and comprehensive analysis.

