# Loan_Default_Prediction
This model predicts loan repayments and defaults. 

I built a neural network to predict loan defaults. I threw it together using Python, TensorFlow/Keras, Pandas and Scikit-learn. It's a solid portfolio piece showing how to handle real-world messy data and crank out useful predictions in finance.

## Project Overview
This script analyzes historical loan data to forecast if applicants will default. It's got 307k samples and 122 features—stuff like income, credit amounts, and demographics. The big challenge is that it has imbalanced classes (only 8% defaults) and tons of missing values. The code has 8 sections: loading data, handling nulls, checking balance, SMOTE oversampling, plotting, encoding, training a neural net, and evaluating with sensitivity and AUC-ROC. I even added threshold optimization to boost default detection, because in banking, predicting defaults is more important than predicting repayments.

## How It Works
The code is structured in clear, commented steps—easy to follow or modify. Here's the breakdown:

### Imports & Setup: 
Grabs libraries for data handling (Pandas/Numpy), viz (Matplotlib/Seaborn), preprocessing (Scikit-learn), balancing (Imbalanced-learn), and DL (TensorFlow/Keras). Prints TF version to confirm everything's good.

### Data Loading & Cleaning: 
Loads CSVs, drops useless IDs, checks/prints TARGET distribution (imbalance alert!), scans for nulls (up to 69% in some columns), plots top missings, drops high-null cols if needed (threshold 70%), and imputes the rest—medians for numbers, modes for categories. Ends with zero nulls.

### Imbalance Check: 
Calculates and prints default/repay percentages, plots a bar chart. Highlights the 11:1 skew—key for why models flop without balancing.

### Balancing with SMOTE: 
Splits data (80/20 stratified), label-encodes categories temporarily, applies SMOTE to oversample defaults in training only (avoids leakage). Prints before/after distributions.

### Visualization: 
Side-by-side count plots show the imbalance fix—original's lopsided, post-SMOTE's even.

### Feature Prep: 
Uses ColumnTransformer to scale numerics and one-hot encode categories (drops first for no multicollinearity). Expands to ~244 features, ready for the NN.

### Neural Network: 
Builds an MLP (128-64-32-1 layers, ReLU, dropout for regularization). Compiles with Adam and binary cross-entropy, trains with early stopping (patience 5). Predicts probs, thresholds to binary, calculates sensitivity (recall for defaults), prints report, plots confusion matrix.

### Evaluation: 
Computes AUC-ROC (~0.73), plots ROC curve and loss/accuracy histories. 

### Bonus: 
Optimizes threshold via precision-recall curve, jumping sensitivity from ~0.03 to ~0.46—huge for catching more defaults.

Runtime: 20-30 mins on CPU (SMOTE and training are the bottlenecks). Use a GPU for speed.

## Why This Is Valuable and Useful
In finance, predicting defaults isn't just academic—it's about saving banks millions by spotting risky loans early without rejecting good ones. My model hits a solid AUC-ROC (0.73), meaning it discriminates well, but the real value is in business tweaks like threshold optimization: base sensitivity sucks (~3% defaults caught), but tuned, it's ~46%, balancing risk vs. opportunity. This project's useful for anyone in fintech/ML: it's a template for imbalanced binary classification, showing end-to-end from raw data to interpretable predictions. As a portfolio piece, it highlights skills in DL pipelines, data engineering, and business-aligned ML. Fork it for your own datasets, or experiment with more layers/class weights.
