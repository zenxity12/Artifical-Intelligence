import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("student_performance.csv")

# Handle missing values (forward fill)
data.ffill(inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target variable
X = data.drop('final_grade', axis=1)
y = data['final_grade']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree
clf_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_tree.fit(X_train, y_train)

# Predict and evaluate for Decision Tree
y_pred_tree = clf_tree.predict(X_test)
print("Decision Tree Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Precision:", precision_score(y_test, y_pred_tree))
print("Recall:", recall_score(y_test, y_pred_tree))
print("F1-Score:", f1_score(y_test, y_pred_tree))

# Hyperparameter tuning for SVM with GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
stratified_cv = StratifiedKFold(n_splits=3)  # Use 3-fold cross-validation
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=stratified_cv)
grid.fit(X_train_scaled, y_train)

# Best parameters for SVM
print("Best Parameters for SVM:", grid.best_params_)

# Predict and evaluate for SVM
y_pred_svm = grid.best_estimator_.predict(X_test_scaled)
print("SVM Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1-Score:", f1_score(y_test, y_pred_svm))

# Metrics for visualization
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
dt_scores = [accuracy_score(y_test, y_pred_tree), precision_score(y_test, y_pred_tree), recall_score(y_test, y_pred_tree), f1_score(y_test, y_pred_tree)]
svm_scores = [accuracy_score(y_test, y_pred_svm), precision_score(y_test, y_pred_svm), recall_score(y_test, y_pred_svm), f1_score(y_test, y_pred_svm)]

# Plotting the comparison
x = range(len(metrics))
plt.bar(x, dt_scores, width=0.4, label='Decision Tree', align='center')
plt.bar([p + 0.4 for p in x], svm_scores, width=0.4, label='SVM', align='center')
plt.xticks([p + 0.2 for p in x], metrics)
plt.legend()
plt.title('Algorithm Performance Comparison')
plt.show()
