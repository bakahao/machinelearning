import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

# Load the dataset
file_path = 'C:/Users/shenhao/OneDrive/Inti/Degree/Sem 7/Machine Learning/Preprocessed_Tesla_Dataset.csv'
tesla_data = pd.read_csv(file_path)

# Feature Engineering
# Calculate rolling averages and volatility using Adj Close
tesla_data['3_day_avg'] = tesla_data['Adj Close'].rolling(window=3).mean()
tesla_data['7_day_avg'] = tesla_data['Adj Close'].rolling(window=7).mean()
tesla_data['volatility'] = tesla_data['High'] - tesla_data['Low']
tesla_data['price_change'] = tesla_data['Adj Close'] - tesla_data['Adj Close'].shift(1)

# Add Momentum Indicators
# Calculate RSI using Adj Close
tesla_data['RSI'] = tesla_data['Adj Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
    tesla_data['Adj Close'].diff().abs().rolling(window=14).mean()
tesla_data['RSI'] = tesla_data['RSI'] * 100  # Convert RSI to a percentage

# Calculate MACD using Adj Close
tesla_data['MACD'] = tesla_data['Adj Close'].ewm(span=12, adjust=False).mean() - \
                     tesla_data['Adj Close'].ewm(span=26, adjust=False).mean()

# Calculate OBV using Adj Close
tesla_data['OBV'] = (np.sign(tesla_data['Adj Close'].diff()) * tesla_data['Volume']).cumsum()

# Drop rows with NaN values created by rolling windows
tesla_data.dropna(inplace=True)

# Save the feature-engineered dataset
tesla_data.to_csv('Feature_Engineered_Tesla_Dataset.csv', index=False)

# Define a threshold for significant changes
threshold = 0.01  # 1% threshold

# Create a target column indicating significant trends based on Adj Close
tesla_data['Percentage_Change'] = (tesla_data['Adj Close'].pct_change().shift(-1)) * 100
tesla_data['Target'] = tesla_data['Percentage_Change'].apply(
    lambda x: 2 if x > threshold else (1 if x < -threshold else 0)
)

# Drop the last row as it will have an undefined target
tesla_data = tesla_data[:-1]

# Select features for modeling (using Adj Close-based features)
features = ['Open', 'High', 'Low', 'Volume', 'DayOfWeek', '3_day_avg', '7_day_avg', 'volatility', 
            'price_change', 'RSI', 'MACD', 'OBV']
X = tesla_data[features]
y = tesla_data['Target']

# Standardize the features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split the balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

#XGBoost Before Tuning
# Define and train the XGBoost model before tuning
xgb_before = xgb.XGBClassifier(
    random_state=42,
    eval_metric='mlogloss'  # Multi-class log-loss
)
xgb_before.fit(X_train, y_train)

# Make predictions with the untuned XGBoost model
xgb_preds_before = xgb_before.predict(X_test)

# Evaluate the untuned XGBoost model
xgb_report_before = classification_report(y_test, xgb_preds_before, zero_division=0, target_names=["No Change (0)", 
                                                                                                   "Significant Fall (1)", 
                                                                                                   "Significant Rise (2)"])
xgb_accuracy_before = accuracy_score(y_test, xgb_preds_before)

print("\nXGBoost Results (Untuned)")
print("Accuracy:", xgb_accuracy_before)
print(xgb_report_before)



# XGBoost After Tuning
# Hyperparameter tuning using RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1 regularization
    'reg_lambda': [0.5, 1.0, 1.5, 2.0]  # L2 regularization
}

random_search = RandomizedSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
    param_distributions=param_dist,
    scoring='accuracy',
    n_iter=50,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Perform hyperparameter tuning
random_search.fit(X_train, y_train)

# Best parameters from tuning
best_params = random_search.best_params_
print("\nBest Parameters from Tuning:", best_params)

# Train XGBoost with the best parameters
xgb_after = xgb.XGBClassifier(**best_params, random_state=42, eval_metric='mlogloss')
xgb_after.fit(X_train, y_train)

# Make predictions with the tuned XGBoost model
xgb_preds_after = xgb_after.predict(X_test)

# Evaluate the tuned XGBoost model
xgb_report_after = classification_report(y_test, xgb_preds_after, zero_division=0, target_names=["No Change (0)", "Significant Fall (1)", "Significant Rise (2)"])
xgb_accuracy_after = accuracy_score(y_test, xgb_preds_after)

print("\nXGBoost Results (Tuned)")
print("Accuracy:", xgb_accuracy_after)
print(xgb_report_after)



# Random Forest Before Tuning
# Train a Random Forest Classifier with default settings
rf_clf_untuned = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
rf_clf_untuned.fit(X_train, y_train)

# Make predictions on the test set for Random Forest (untuned)
rf_clf_preds_untuned = rf_clf_untuned.predict(X_test)

# Evaluate the untuned Random Forest model
rf_clf_report_untuned = classification_report(y_test, rf_clf_preds_untuned, zero_division=0, target_names=["No Change (0)", 
                                                                                                           "Significant Fall (1)", 
                                                                                                           "Significant Rise (2)"])
rf_clf_accuracy_untuned = accuracy_score(y_test, rf_clf_preds_untuned)

print("\nRandom Forest Results (Untuned)")
print("Accuracy:", rf_clf_accuracy_untuned)
print(rf_clf_report_untuned)



#Random Forest After Tuning
# Define the parameter grid for tuning
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Perform RandomizedSearchCV for hyperparameter tuning
rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=param_dist,
    scoring='accuracy',
    n_iter=50,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit the randomized search
rf_random_search.fit(X_train, y_train)

# Best parameters from tuning
best_params_rf = rf_random_search.best_params_
print("\nBest Parameters from Tuning:", best_params_rf)

# Train Random Forest with the best parameters
rf_clf_tuned = RandomForestClassifier(**best_params_rf, random_state=42, class_weight='balanced')
rf_clf_tuned.fit(X_train, y_train)

# Make predictions on the test set for Random Forest (tuned)
rf_clf_preds_tuned = rf_clf_tuned.predict(X_test)

# Evaluate the tuned Random Forest model
rf_clf_report_tuned = classification_report(y_test, rf_clf_preds_tuned, zero_division=0, target_names=["No Change (0)", 
                                                                                                       "Significant Fall (1)", 
                                                                                                       "Significant Rise (2)"])
rf_clf_accuracy_tuned = accuracy_score(y_test, rf_clf_preds_tuned)

print("\nRandom Forest Results (Tuned)")
print("Accuracy:", rf_clf_accuracy_tuned)
print(rf_clf_report_tuned)

# Plot the confusion matrix
conf_matrix_before = confusion_matrix(y_test, xgb_preds_before)
class_names = ["No Change (0)", "Fall (1)", "Rise (2)"]
disp_before = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_before, display_labels=class_names)
disp_before.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix for Untuned XGBoost")

# Plot the confusion matrix
conf_matrix_after = confusion_matrix(y_test, xgb_preds_after)
disp_after = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_after, display_labels=class_names)
disp_after.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix for Tuned XGBoost")

# Plot the confusion matrix for untuned model
conf_matrix_untuned = confusion_matrix(y_test, rf_clf_preds_untuned)
class_names = ["No Change (0)", "Fall (1)", "Rise (2)"]
disp_untuned = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_untuned, display_labels=class_names)
disp_untuned.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix for Untuned Random Forest")

# Plot the confusion matrix for tuned model
conf_matrix_tuned = confusion_matrix(y_test, rf_clf_preds_tuned)
disp_tuned = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_tuned, display_labels=class_names)
disp_tuned.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix for Tuned Random Forest")

# Plot the class distribution before balancing
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
y.value_counts().plot(kind='bar', color=['orange', 'blue', 'green'], alpha=0.7)  # Add a color for the third class
plt.title("Class Distribution Before Balancing")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1, 2], labels=["No Change (0)", "Fall (1)", "Rise (2)"], rotation=0)

# Plot the class distribution after balancing
plt.subplot(1, 2, 2)
y_balanced.value_counts().plot(kind='bar', color=['orange', 'blue', 'green'], alpha=0.7)  # Add a color for the third class
plt.title("Class Distribution After Balancing")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(ticks=[0, 1, 2], labels=["No Change (0)", "Fall (1)", "Rise (2)"], rotation=0)

# Binarize the output for multi-class
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
y_pred_proba = xgb_after.predict_proba(X_test)

# Define a dictionary to store ROC data for each class
roc_data = {}

# Calculate ROC curve and AUC for each class
for i in range(3):  # For each class
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
    auc_score = auc(fpr, tpr)
    roc_data[i] = {"fpr": fpr, "tpr": tpr, "auc": auc_score}

# Plot individual ROC curves for each class
for class_id in range(3):
    plt.figure(figsize=(8, 6))
    plt.plot(
        roc_data[class_id]["fpr"],
        roc_data[class_id]["tpr"],
        label=f"Class {class_id} (AUC = {roc_data[class_id]['auc']:.2f})",
        color='blue',
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)  # Baseline
    plt.title(f"ROC Curve for Class {class_id}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.5)


# Show the plots
plt.tight_layout()
plt.show()

