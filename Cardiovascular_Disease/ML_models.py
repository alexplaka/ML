import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def get_cv_score(X, y, Classifier, /, options={}, *, cv=10, scoring='accuracy'):
    scores = cross_val_score(Classifier(**options), X, y, cv=cv, scoring=scoring)

    return scores.mean()


def svc_model(X_train, X_test, y_train, y_test):
    """
    Support Vector Machine Classifier model: choose kernel.
    Then fit to data and predict.
    Return model and classification report.
    """

    kernel = 'linear'  # can try 'rbf' --> getting similar results.
    clf = SVC(kernel=kernel, probability=True)
    model = clf.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict classes in test set
    print(f'Estimator: Support Vector Machine Classifier (kernel={kernel})\n',
          classification_report(y_test, y_pred))

    return model, y_pred, classification_report(y_test, y_pred, output_dict=True)


def kNN_model(X_train, X_test, y_train, y_test):
    """
    K-nearest neighbors model: find best hyperparameter k and weight method.
    Then fit to (unscaled) data and predict.
    Return model and classification report.
    """

    scores_u = []  # uniform weights
    scores_d = []  # weights inversely related to distance
    k_range = range(20, 71)
    for k in k_range:
        for weight in ['uniform', 'distance']:
            score = get_cv_score(X_train, y_train, KNeighborsClassifier,
                                 {'n_neighbors': k, 'weights': weight})
            scores_u.append(score) if weight == 'uniform' else scores_d.append(score)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    ax.plot(k_range, scores_u)
    ax.plot(k_range, scores_d)
    ax.set(title='K-Nearest Neighbors: Mean (cv=10) Accuracy Scores\n'
                 'for Different Weight Methods',
           xlabel='$k$ neighbors', ylabel='Accuracy')
    ax.legend(['uniform', 'distance'], loc='lower right')
    fig.savefig('kNN_cv_search.png')

    k_best_u = scores_u.index(np.max(scores_u))
    k_best_d = scores_d.index(np.max(scores_d))
    print(f"Estimator: K-nearest neighbors:\n"
          f"Optimal k={k_best_u + k_range[0]} with an accuracy of {scores_u[k_best_u]} with "
          "uniform weight method.\n"
          f"Optimal k={k_best_d + k_range[0]} with an accuracy of {scores_d[k_best_d]} with "
          "distance weight method.")

    optimal_hyperparams = {'k': k_best_u + 1 if k_best_u >= k_best_d else k_best_d + 1,
                           'weights': 'uniform' if k_best_u >= k_best_d else 'distance'}

    model = KNeighborsClassifier(n_neighbors=optimal_hyperparams['k'],
                                 weights=optimal_hyperparams['weights'])
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict classes in test set
    print(classification_report(y_test, y_pred))

    return model, y_pred, classification_report(y_test, y_pred, output_dict=True)


def RF_model(X_train, X_test, y_train, y_test):
    """
    Random Forest model model: find best hyperparameters.
    Then fit to data and predict.
    Return model and classification report.
    """

    clf = RandomForestClassifier(random_state=0)

    hyperparams = {"max_features": ["auto"],
                   "max_leaf_nodes": [None],
                   "max_depth": [9]}
    cv = 10
    model = GridSearchCV(clf, hyperparams, cv=cv)
    model.fit(X_train, y_train)
    print(f"\nEstimator: Random Forest Classifier. \n"
          f"Best parameters after grid search with cross-validation (cv={cv}): \n"
          f"{model.best_params_}\nwith score {model.best_score_}")
    # If model.refit is true (default), the model automatically refits to all of X_train.
    print(f"Automatic refit to full X_train: {model.refit}")

    y_pred = model.predict(X_test)  # Predict classes in test set
    print(classification_report(y_test, y_pred))

    return model, y_pred, classification_report(y_test, y_pred, output_dict=True)


def group_model(X_test, X_test_scaled, y_test, models, *, weights=None):
    """
    ** Creating a group/ensemble model of all of the fitted models **
    The below manipulations are equivalent to using sklearn.ensemble.VotingClassifier()
    on the unfitted models with 'soft' voting.
    Since we have already fit the models, calculating the average class probabilities manually.
    """

    n = len(models)
    wgts = np.ones(n) / n if weights is None else weights

    p_weighted_vals = np.empty((X_test.shape[0], 2, n))  # initialize matrix

    # Get probablility values for both classes for each estimator and weigh them.
    for i, model in enumerate(models):
        print(model)
        p_weighted_vals[:, :, i] = wgts[i] * model.predict_proba(X_test if str(model).startswith('K')
                                                                 else X_test_scaled)

    ensemble_p_vals = np.mean(p_weighted_vals, axis=2)
    y_pred = ensemble_p_vals.argmax(axis=1)
    print(classification_report(y_test, y_pred))

    return y_pred, classification_report(y_test, y_pred, output_dict=True)
