import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFECV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report

rng = np.random.RandomState(0)  # initialize random number generator
# %%
# Load the wine dataset: n_samples=178, n_features=13.
data, y = load_wine(return_X_y=True, as_frame=True)
print(data.info())
# %%
# Select best features based on mutual information.
select_features = SelectKBest(score_func=mutual_info_classif, k=11).fit(data, y)

# Plot MI scores
fig, ax = plt.subplots(2, 1, figsize=(6, 10), dpi=100)
ax[0].bar(np.arange(data.columns.shape[0]), select_features.scores_)
ax[0].set_xticks(np.arange(data.shape[1]))
ax[0].set(title='Mutual Information scores for features',
          xlabel='Feature #', ylabel='MI')

# Arbitrary choice: eliminate 2 features with the lowest MI scores.
print("#: FEATURE NAME")
for i, col in enumerate(data.columns):
    print(f'{i}: {col}')
print('\nCan eliminate two features with lowest MI score: ',
      data.columns[2], ', ', data.columns[7], '.', sep='')

del i, col

# Get new dataset (convert to dataframe) with reduced number of features
# X = pd.DataFrame(select_features.transform(data), columns=data.columns.delete([2, 7]))

# Try recursive feature elimination (with cross-validation) using SVM with linear kernel.
clf = SVC(kernel='linear')
rfecv = RFECV(clf, step=1, min_features_to_select=1, cv=5, scoring='accuracy')
rfecv.fit(data, y)
print(f"\nOptimal number of features using RFECV: {rfecv.n_features_}")

# Plot number of features vs. cross-validation scores
ax[1].plot(np.arange(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, '.-r')
ax[1].set(title='Recursive feature elimination using SVM with linear kernel',
          xlabel="Number of features selected", ylabel="Cross validation score (accuracy)")
ax[1].set_xticks(np.arange(rfecv.grid_scores_.shape[0] + 1))

fig.savefig('featureselection.png')
# RFE result: keep all features.
# Same result when two features with low MI were already eliminated.
print('\nKeeping all 13 features.')
# %%
# Split data infor train and test sets.
X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=rng,
                                                    test_size=0.2, stratify=y)

# %%
# Try different estimators
estimators = [GaussianNB(),
              RidgeClassifierCV(alphas=np.logspace(-3, 1, num=10)),
              SVC(kernel='linear'),
              RandomForestClassifier(random_state=rng)]

models = dict()

for estimator in estimators:
    estimator_name = str(estimator)[:str(estimator).index('(')]

    # Make a pipeline
    pipe = make_pipeline(StandardScaler(), estimator)
    # print(pipe.get_params())

    if 'GaussianNB' in estimator_name:
        print("\nEstimator: Gaussian Naive Bayes Classifier.")
        model = pipe.fit(X_train, y_train)
    elif 'Ridge' in estimator_name:
        print("\nEstimator: Ridge Classifier with cross-validation.")
        model = pipe.fit(X_train, y_train)
    elif 'SVC' in estimator_name:
        print("\nEstimator: Support Vector Machine Classifier.")
        model = pipe.fit(X_train, y_train)
    else:
        hyperparams = {"randomforestclassifier__max_features": ["auto", "sqrt"],
                       "randomforestclassifier__max_leaf_nodes": [None, 2, 3, 5],
                       "randomforestclassifier__max_depth": [None, 1, 3]}
        model = GridSearchCV(pipe, hyperparams, cv=10)
        model.fit(X_train, y_train)
        print("\nEstimator: Random Forest Classifier. \n"
              "Best parameters after grid search with cross-validation (cv=10): \n"
              f"{model.best_params_}\nwith score {model.best_score_}")
        # If model.refit is true (default), the model automatically refits to all of X_train.
        print(f"Automatic refit to full X_train: {model.refit}")

    y_pred = model.predict(X_test)  # Predict classes in test set

    # *** Calculate metrics of prediction quality ***
    # print('Matthews correlation coefficient=', matthews_corrcoef(y_test, y_pred))
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Append model to 'models' dict (requires Python 3.9)
    models |= {estimator_name: {'model': model,
                                'y_pred': y_pred,
                                'matthews': matthews_corrcoef(y_test, y_pred),
                                'confusion': confusion_matrix(y_test, y_pred)}
               }

    # Compute learning curve
    lc_sizes, train_scores, cv_scores = learning_curve(pipe, X_train, y_train, cv=5,
                                                       train_sizes=np.linspace(0.1, 1.0, 10),
                                                       scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    cv_scores_std = np.std(cv_scores, axis=1)

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ax.grid()
    ax.plot(lc_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.fill_between(lc_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1, color="r")

    ax.plot(lc_sizes, cv_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.fill_between(lc_sizes, cv_scores_mean - cv_scores_std,
                    cv_scores_mean + cv_scores_std,
                    alpha=0.1, color="g")

    ax.set(title=f"Learning curve for {estimator_name}",
           xlabel="Training examples", ylabel='Accuracy Score')
    ax.legend(loc="lower right")
    fig.savefig(f'{estimator_name}_LearningCurve.png')
# %%
joblib.dump(models, 'wine_ml.joblib')  # save models for later use
# models2 = joblib.load('wine_rf.joblib')  # reload models
