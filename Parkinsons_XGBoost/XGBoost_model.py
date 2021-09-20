# Predict if a patient has Parkinsons disease.
# Using XGBoost classifier on dataset with n_samples=195, n_features=22.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, classification_report)

# %%
data = pd.read_csv("parkinsons.csv", index_col="name")
X = data.copy()
y = X.pop("status")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# %%
train_perc_pos = y_train[y_train == 1].count() / (
    y_train[y_train == 1].count() + y_train[y_train == 0].count())
test_perc_pos = y_test[y_test == 1].count() / (
    y_test[y_test == 1].count() + y_test[y_test == 0].count())
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
model = XGBClassifier()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
score = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
matt = matthews_corrcoef(y_test, y_pred)
print(classification_report(y_test, y_pred))
