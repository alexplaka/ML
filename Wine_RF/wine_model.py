from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report
import joblib

# %%
data, y = load_wine(return_X_y=True, as_frame=(True))

X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=0,
                                                    test_size=0.2, stratify=y)

pipe = make_pipeline(StandardScaler(),
                     RandomForestClassifier(n_estimators=100, random_state=0))
# %%
# print(pipe.get_params())

hyperparams = {"randomforestclassifier__max_features": ["auto", "sqrt", "log2"],
               "randomforestclassifier__max_leaf_nodes": [None, 2, 3, 5, 7, 9],
               "randomforestclassifier__max_depth": [None, 1, 3, 5, 7]}
model = GridSearchCV(pipe, hyperparams, cv=10)
model.fit(X_train, y_train)
print(f"Best parameters after grid search: {model.best_params_}\nwith score {model.best_score_}")

# If model.refit is true (default), the model automatically refits to all of X_train.
print(f"Automatic refit to full X_train: {model.refit}")

y_pred = model.predict(X_test)  # Predict classes in test set
# %%
# Calculate metrics of prediction quality
f1 = f1_score(y_test, y_pred, average='weighted')
matt = matthews_corrcoef(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# %%
joblib.dump(model, 'wine_rf.joblib')  # save model for later use
# model2 = joblib.load('wine_rf.pkl')  # reload model
