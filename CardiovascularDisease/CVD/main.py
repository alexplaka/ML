import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import preprocessor
import data_visualizer
import ML_models
import postprocessor
import joblib

rng = np.random.RandomState(0)  # initialize random number generator


def load():
    """ Import data and rename the columns """

    df = pd.read_csv("../cardio_data.csv", index_col="id")

    # Improve column names for readability
    df.rename(str.capitalize, axis='columns', inplace=True)
    df.rename(columns={'Ap_hi': 'BP_hi', 'Ap_lo': 'BP_lo', 'Gluc': 'Glucose', 'Alco': 'Alcohol'},
              inplace=True)

    return df


if __name__ == "__main__":
    data_orig = load()

    # add new features BMI (float) and Overweight (0/1)
    data = preprocessor.add_features(data_orig)
    data_clean = preprocessor.cleanup(data)

    # Categorical and numeric features
    cat_feats = ["Gender", "Overweight", "Cholesterol", "Glucose", "Smoke", "Alcohol", "Active"]
    num_feats = ["Age", "Height", "Weight", "BMI", "BP_hi", "BP_lo"]

    # Draw plots to describe the data.
    fig_cat = data_visualizer.draw_cat_plot(data_clean, "Cardio", cat_feats,
                                            output_filename='catfeats_plot.png')
    fig_corr, corr, target_scores = data_visualizer.draw_corr_matrix(data_clean)

    X = data_clean.copy()
    y = X.pop("Cardio")  # target variable

    # Split data (with all of the features) into train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

    # Use only selected features with highest correlation or MI to target
    # X1 = X.drop(columns=["Gender", "Height", "Smoke", "Alcohol"])
    # X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state=rng)
    # Result: only using these features does not improve model accuracy.

    # Use custom scaler (gives same results as standard scaler)
    X_train_scaled, X_test_scaled = preprocessor.scaler(X_train, X_test,
                                                        cat_feats=cat_feats,
                                                        num_feats=num_feats)

    svc_model, svc_pred, svc_report = ML_models.svc_model(X_train_scaled, X_test_scaled,
                                                          y_train, y_test)
    kNN_model, kNN_pred, kNN_report = ML_models.kNN_model(X_train, X_test,
                                                          y_train, y_test)
    RF_model, RF_pred, RF_report = ML_models.RF_model(X_train_scaled, X_test_scaled,
                                                      y_train, y_test)

    # Note on model quality up to here:
    # Models do not give an accuracy greater than 72%.

    # * Will try combining the three models *
    # Giving more weight to Random Forest model. (Using equal weights gives similar results.)
    group_pred, group_report = ML_models.group_model(X_test, X_test_scaled, y_test,
                                                     [svc_model, kNN_model, RF_model],
                                                     weights=(0.32, 0.32, 0.36))

    print(f"Model\t\t\t\tAccuracy\n"
          f"-----\t\t\t\t--------\n"
          f"SVC\t\t\t\t\t{svc_report['accuracy']:.3f}\n"
          f"kNN\t\t\t\t\t{kNN_report['accuracy']:.3f}\n"
          f"RF\t\t\t\t\t{RF_report['accuracy']:.3f}\n"
          f"Model Ensemble\t\t{group_report['accuracy']:.3f}")

    all_preds = [svc_pred, kNN_pred, RF_pred, group_pred]

    common_mislabels, common_fp, common_fn = postprocessor.model_group_mislabels(y_test,
                                                                                 *all_preds)

    data_mislabeled = X.loc[common_mislabels, :].copy()
    data_mislabeled.loc[common_fp, "Mislabel"] = "False Positive"  # new column: false positive
    data_mislabeled.loc[common_fn, "Mislabel"] = "False Negative"  # false negative
    fig_cat2 = data_visualizer.draw_cat_plot(data_mislabeled, "Mislabel", cat_feats,
                                             output_filename="catplot_mislabels.png")

    # Random forest model gives the best accuracy at 72%. Saving it for future use.
    joblib.dump(RF_model, "RF_model.joblib")
