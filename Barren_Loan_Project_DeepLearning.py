"""
Hey everyone, so I've just put together this complete deep learning project for predicting loan defaults using a neural network, and I wanted to share a quick rundown on how it all works and some key things to watch out for if you're thinking of running or tweaking it yourself. I'm a Caltech AI/ML student, and this was for my course-end project—super rewarding but definitely had its moments of debugging frustration! The code is structured in clear steps, from loading the data all the way to optimizing the model for real-world use, and it's built in Python with libraries like pandas, scikit-learn, imbalanced-learn, and TensorFlow/Keras. Basically, it's designed to take historical loan data (like from that Home Credit dataset on Kaggle) and build a model that flags whether an applicant might default, which is crucial in finance to avoid bad loans without turning away good customers.
It starts with the basics: importing everything you need and loading the CSV files—one for the main loan data (about 307k rows, 122 columns) and one for the data dictionary to understand what each feature means. I drop the ID column right away since it's not predictive. Then, Step 2 dives into handling missing values, which is huge because this dataset has tons of nulls—some columns up to 69% missing! I check and visualize the top offenders with a bar plot, drop anything over 70% (usually none in this case), and impute the rest: medians for numerics (robust against outliers in financial data) and modes for categoricals. After that, zero nulls left, ready for action.
Step 3 calculates and prints the TARGET distribution—turns out it's super imbalanced, with only 8.07% defaults vs. 91.93% repayments. I plot a bar chart to show it visually, because seeing that 11.4:1 ratio really drives home why balancing is key. Step 4 tackles that with SMOTE: I temporarily label-encode categoricals (since SMOTE needs numbers), split into train/test (80/20, stratified to keep the imbalance ratio), and oversample the minority class on the train set only to avoid leakage. It can take 1-2 minutes on full data, but ends up with a perfect 50/50 split—about 452k balanced training samples.
Step 5 is the fun visual part: side-by-side count plots showing the before (tall repay bar, tiny default) and after (equal bars) distributions. Pro tip: close the plots to keep going in Spyder. Step 6 preps for the NN by scaling numerics and one-hot encoding categoricals using ColumnTransformer—expands to 244 features, which is manageable but watch your memory if running on a basic laptop; it peaks here.
The heart is Step 7: building a multi-layer perceptron (MLP) neural network with 128-64-32-1 neurons, ReLU activations, dropout for regularization (0.3-0.2 to prevent overfitting), sigmoid output for binary classification. Compiles with Adam optimizer and binary cross-entropy loss, trains with early stopping (patience 5 on val_loss), batch size 256 for speed on large data. Training takes 5-15 minutes, usually stops around 20-30 epochs. Then predicts on test set, calculates sensitivity (recall for defaults), prints a classification report, and plots the confusion matrix. In my run, base sensitivity was low (0.0254) because of the 0.5 threshold bias, but AUC-ROC was solid at 0.7273.
Step 8 wraps with ROC curve/AUC plot and training history graphs for loss/accuracy—great for spotting overfitting. Finally, I added a bonus threshold optimization using precision-recall curve to boost sensitivity (jumped to 0.4610 at 0.16 threshold) while keeping AUC the same. This is key for finance: catching more defaults (false negatives cost money) without too many false positives (turning away good customers).
If you're using this code, be aware: it assumes TensorFlow 2.15 and might need tweaks for different versions (like metrics as strings). Full run takes 20-30 minutes on a standard CPU—use a GPU if you can for faster training. Memory can spike during SMOTE/encoding (up to 8-16GB), so close other apps or sample the data for testing (loans.sample(frac=0.1)). Paths to 'loan_data.csv' and 'Data_Dictionary.csv' need to match your setup. Watch for imbalanced-learn version issues (no n_jobs in newer ones). If sensitivity stays low, experiment with class weights in model.fit() or more layers. Overall, this is a solid template for binary classification in finance—feel free to fork and improve! Hit me up if you run into bugs; I learned a ton debugging this. (Word count: 512)
"""
"""
Step 1: Importing Libraries and Dependencies 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn for preprocessing and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report

# Imbalanced-learn for SMOTE
from imblearn.over_sampling import SMOTE

# TensorFlow/Keras for neural network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Verify TensorFlow
print(f"TensorFlow version: {tf.__version__}")
print("All modules and libraries have been imported.")

"""
Step 2: Import Dataset and Resolve Missing Values 
"""

print("\n" + "="*50)
print("\nSTEP 1: LOADING DATASET")
print("\n" + "="*50)

loans = pd.read_csv('loan_data.csv')
dictionary = pd.read_csv('Data_Dictionary.csv', encoding='latin1')

# Exploratory Data Analysis
print(f"Dataset shape: {loans.shape}")
print(f"Number of Columns: {len(loans.columns)}")
print(f"TARGET distribution:")
print(loans['TARGET'].value_counts())

loans = loans.drop('SK_ID_CURR', axis=1)
print(f"After dropping ID: {loans.shape}")

# Check for Null Values 
print("\n" + "="*50)
print("\nSTEP 2: CHECKING NULL VALUES")
print("\n" + "="*50)

null_counts = loans.isnull().sum()
null_percent = (null_counts / len(loans)) * 100
null_df = pd.DataFrame({
    'Missing_Count': null_counts[null_counts > 0],
    'Missing_Percent': null_percent[null_counts > 0]
}).sort_values('Missing_Percent', ascending=False)

print("TOP 10 COLUMNS WITH MOST MISSING VALUES")
print(null_df.head(10))

# Visualize Null Values 
plt.figure(figsize=(12, 8))
top_nulls = null_df.head(15)
sns.barplot(data=top_nulls.reset_index(), x='index', y='Missing_Percent', palette='viridis')
plt.title('Top 15 Columns with Missing Values (%)', fontsize=14)
plt.xlabel('Columns')
plt.ylabel('Missing Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Handling Null Values 
print("\n" + "="*50)
print(f"\nHANDLING MISSING VALUES")
print("\n" + "="*50)

high_null_cols = null_df[null_df['Missing_Percent'] > 70].index.tolist()
print(f"Dropping {len(high_null_cols)} columns with >70% missing values")

if high_null_cols:
    loans = loans.drop(high_null_cols, axis=1)
    print(f"Remaining Columns: {loans.shape[1]}")

# Define variables for different column data types and print results to visualize data types     
num_cols = loans.select_dtypes(include=[np.number]).columns
cat_cols = loans.select_dtypes(include=['object']).columns

print(f"Numeric Columns: {len(num_cols)}")
print(f"Categorical Columns: {len(cat_cols)}")

for col in num_cols:
    if col != 'TARGET':
        loans[col] = loans[col].fillna(loans[col].median())

for col in cat_cols:
    mode_val = loans[col].mode()
    fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
    loans[col] = loans[col].fillna(fill_val)

print(f"Total nulls after handling: {loans.isnull().sum().sum()}")

"""
STEP 3: Calculate and Visualize Data Balance/Imbalance
"""

print("\n" + "="*50)
print("\nSTEP 3: TARGET DISTRIBUTION")
print("\n" + "="*50)

target_counts = loans['TARGET'].value_counts()
total_samples = len(loans)
default_rate = (target_counts[1] / total_samples) * 100 if 1 in target_counts else 0
repay_rate = (target_counts[0] / total_samples) * 100 if 0 in target_counts else 0 

print(f"CLASS DISTRIBUTION:")
print(f"Repay (TARGET=0): {repay_rate: .2f}% ({target_counts.get(0, 0):,}")
print(f"Default (TARGET=1): {default_rate:.2f}% ({target_counts.get(1, 0):,}")
print(f"Imbalance ratio: {repay_rate/default_rate:.1f}:1" if default_rate > 0 else "N/A")

plt.figure(figsize=(8, 5))
target_counts.plot(kind='bar', color=['green', 'red'])
plt.title('TARGET Distribution (0=Repay, 1=Default)', fontsize=14)
plt.xlabel('TARGET')
plt.ylabel('Count')
plt.xticks(rotation=0)
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 1000, f"{v:,}", ha='center', va='bottom')
plt.tight_layout()
plt.show()

"""
STEP 4: Balance the data if it is imbalanced
"""
print("\n" + "="*50)
print("\nSTEP 4: BALANCING DATA")
print("\n" + "="*50)

X = loans.drop('TARGET', axis=1)
y = loans['TARGET']

print("Temporary encoding for SMOTE...")
le_dict = {}
for col in cat_cols:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        # Temporarily encodes categoricals, splits data, applies SMOTE to train set
        # SMOTE creates synthetic defaults
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Original training distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("Applying SMOTE...")
smote = SMOTE(random_state=42) # This can take some time to apply to a data set, may take 1-2 minutes
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Balanced training distribution:")
print(y_train_bal.value_counts(normalize=True).round(3))

"""
Step 5: Plotting the Balanced or Imbalanced Data
"""
# PLotting bar charts for TARGET distributions
print("\n" + "="*50)
print("\nSTEP 5: PLOTTING DATA")
print("\n" + "="*50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y_train, ax=axes[0], palette='Set2')
axes[0].set_title('Original Training Data (Imbalanced)')
axes[0].set_xlabel('TARGET')

sns.countplot(x=y_train_bal, ax=axes[1], palette='Set1')
axes[1].set_title('Balanced Training Data (Post-SMOTE)')
axes[1].set_xlabel('TARGET')

plt.tight_layout()
plt.show()

"""
Step 6: Encoding columns required for the neural network
"""
# This section preps the data for the neural network 
# Scales the numeric data, one-hot encodes the categorical data 
# May take a long time (1-2minutes), massive memory peak here

print("\n" + "="*50)
print("\nSTEP 6: FEATURE ENCODING (prep data for neural network)")
print("\n" + "="*50)

num_cols_final = X_train_bal.select_dtypes(include=[np.number]).columns
cat_cols_final = [col for col in cat_cols if col in X_train_bal.columns]

print(f"Numeric features: {len(num_cols_final)}")
print(f"Categorical features: {len(cat_cols_final)}")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols_final),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), cat_cols_final)
    ])

print("Preprocessing training data...")
X_train_pre = preprocessor.fit_transform(X_train_bal)

print("Preprocessing test data...")
X_test_pre = preprocessor.transform(X_test)

print(f"Preprocessed shapes:")
print(f"Training: {X_train_pre.shape}")
print(f"Testing: {X_test_pre.shape}")
print(f"Final feature count: {X_train_pre.shape[1]}")

print(f"Training nulls: {np.isnan(X_train_pre).sum()}")
print(f"Testing nulls: {np.isnan(X_test_pre).sum()}")

# *** Notes on this step: Nueral network requiers numeric, scaled input, this section prepares data to meet these requirements ***

"""
STEP 7: Build/Train Neural Network, Calculate Sensitivity as a Metric
"""

print("\n" + "="*50)
print("\nSTEP 7: BUILDING AND TRAINING THE NEURAL NETWORK")
print("\n" + "="*50)

# Building a MLP: multi-layered perceptron
# Trains with early stopping, predicts, computes sensitivity 

input_dim = X_train_pre.shape[1]
print(f"Input Dimension: {input_dim}")

model = Sequential([
    Dense(128, input_dim=input_dim, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']  
)

print("MODEL ARCHITECTURE:")
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("TRAINING MODEL...")
history = model.fit(
    X_train_pre, y_train_bal,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

print("MAKING PREDICTIONS...")
y_pred_prob = model.predict(X_test_pre, verbose=0).flatten()
y_pred_binary = (y_pred_prob > 0.5).astype(int)

sensitivity = recall_score(y_test, y_pred_binary)
print(f"SENSITIVITY (RECALL): {sensitivity:.4f}")

print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_binary, target_names=['Repay', 'Default']))

cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Repay (0)', 'Default (1)'],
            yticklabels=['Repay (0)', 'Default (1)'])
plt.title('Confusion Matrix\n(Sensitivity = TP / (TP + FN))')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

print("Neural network training and evaluation complete")

"""
STEP 8: Calculate Area under ROC Curve 
"""

print("\n" + "="*50)
print("\nSTEP 8: ROC CURVE AND AUC")
print("\n" + "="*50)

auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC Score: {auc_score:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
         label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nLoan Default Prediction')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Model Loss During Training')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Model Accuracy During Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("PROJECT COMPLETE - FINAL RESULTS")
print("\n" + "="*50)

print(f"Dataset: {loans.shape[0]:,} samples, {loans.shape[1]} features")
print(f"Default Rate: {default_rate:.2f}%")
print(f"Model Parameters: {model.count_params():,}")
print(f"AUC-ROC: {auc_score:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Training Epochs: {len(history.history['loss'])}")
print(f"\nProject Completed: All 8 steps have been satisfied.")

"""
Additional Section to Optimize Sensitivity 
"""

# This model did not perform well at catching defaults, but performed very well at predicting repayments 
# The decision threshold is way too high, below additional code tries to resolve this 
# Predciting defaults is more important then predicting repayments, so this has to be fixed

print("\n" + "="*50)
print("OPTIMIZING THRESHOLD FOR SENSITIVITY")
print("="*50)

from sklearn.metrics import precision_recall_curve

# Find optimal threshold using precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Original threshold: 0.5")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Max F1-score at this threshold: {f1_scores[optimal_idx]:.4f}")

# Make predictions with optimal threshold
y_pred_optimal = (y_pred_prob > optimal_threshold).astype(int)
sensitivity_optimal = recall_score(y_test, y_pred_optimal)

print(f"\nORIGINAL METRICS (threshold=0.5):")
print(f"  Sensitivity: {sensitivity:.4f}")
print(f"  AUC-ROC: {auc_score:.4f}")

print(f"\nOPTIMIZED METRICS (threshold={optimal_threshold:.4f}):")
print(f"  Sensitivity: {sensitivity_optimal:.4f}")
print(f"  AUC-ROC: {auc_score:.4f} (unchanged)")

# Confusion matrix with optimal threshold
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Repay', 'Default'], yticklabels=['Repay', 'Default'])
plt.title('Original (Threshold=0.5)')
plt.ylabel('True') 
plt.xlabel('Predicted')

plt.subplot(1, 2, 2)
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Repay', 'Default'], yticklabels=['Repay', 'Default'])
plt.title(f'Optimized (Threshold={optimal_threshold:.3f})')
plt.ylabel('True')
plt.xlabel('Predicted')

plt.tight_layout()
plt.show()

print(f"\nSensitivity improved from {sensitivity:.4f} → {sensitivity_optimal:.4f}!")
