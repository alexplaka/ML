import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def add_features(df):
    """ Add 'BMI' (float) and 'Overweight' (0/1) columns """

    # Calculate bmi (kg/m^2)
    df["BMI"] = df["Weight"] / (df["Height"] / 100) ** 2
    df["Overweight"] = df["BMI"].apply(lambda x: 0 if x <= 25 else 1)
    # Two more ways of doing the same:
    # df["Overweight"] = pd.factorize(df["BMI"] > 25)[0]
    # df['Overweight'] = pd.factorize(pd.cut(df["BMI"],
    #                                        bins=pd.IntervalIndex.from_tuples([(0, 25), (25,100)])))

    # New columns are added at the end of dataframe. For consistency, make the target column
    # ('Cardio') the last column and reposition new columns.
    df = df[['Age', 'Gender', 'Height', 'Weight', 'BMI', 'Overweight', 'BP_hi', 'BP_lo',
             'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active', 'Cardio']]

    return df


def cleanup(df):
    """
    - Change "Age" from days to years.
    - Make "Gender" values start from 0.
    - Normalize data by making 0 always good and 1 always bad.
    - Remove wrong data or outliers.
    """
    df["Age"] = df["Age"] / 365
    df["Gender"] = df["Gender"] - 1

    # If the value of 'Cholesterol' or 'Glucose' is 1, make the value 0.
    # If the value is more than 1, make the value 1.
    for feat in ["Glucose", "Cholesterol"]:
        df[feat] = df[feat].apply(lambda x: 0 if x == 1 else 1)

    # Clean up data
    df = df[(df.BP_lo < df.BP_hi) &  # Diastolic pressure MUST be less than the systolic
            (df.BP_lo.between(50, 120)) &  # Diastolic blood pressure should be positive
            (df.BP_hi.between(100, 200)) &  # Reasonable values for systolic blood pressure
            (df.Gender < 2) &  # Gender can only be in [0, 1] = [Female, Male]
            # keep heights >= 0.25th percentile (h>=140cm)
            (df.Height >= df.Height.quantile(0.0025)) &
            # keep heights <= 99.75th percentile (h<=198cm)
            (df.Height <= df.Height.quantile(0.9999)) &
            # keep weights >= 0.25th percentile (w>=42kg=92.4lbs)
            (df.Weight >= df.Weight.quantile(0.0025)) &
            # keep weights <= 99.75th percentile (w<=133kg=292.6lbs)
            (df.Weight <= df.Weight.quantile(0.9975)) &
            # keep bmi >= 0.1th percentile (bmi>=16.1)
            (df.BMI >= df.BMI.quantile(0.001)) &
            (df.BMI <= df.BMI.quantile(0.999))]   # keep bmi <= 99.9th percentile (bmi<=59.2)

    return df


def scaler(X_train, X_test, *, cat_feats=[], num_feats=[]):
    """
    Choose between scaling the data using the StandardScaler or heterogeneous scaling of features.
    For heterogeneous scaling, the categorical and numerical features must be specified.
    Then, apply:
    - min_max scaling (from -1 to 1) for categoricals
      (to avoid non-symmetric scaling about zero due to category frequencies in dataset)
    - standard scaling for numerical features
    """

    X_train_scaled = pd.DataFrame()
    X_test_scaled = pd.DataFrame()

    std_scaler = StandardScaler()

    if len(cat_feats) == 0:
        X_train_scaled = std_scaler.fit_transform(X_train)
        X_test_scaled = std_scaler.transform(X_test)
    else:
        mm_scaler = MinMaxScaler(feature_range=(-1, 1))

        for cfeat in cat_feats:
            X_train_scaled[cfeat] = mm_scaler.fit_transform(np.array(X_train[cfeat]).reshape(-1, 1)).squeeze()
            X_test_scaled[cfeat] = mm_scaler.transform(np.array(X_test[cfeat]).reshape(-1, 1)).squeeze()

        for nfeat in num_feats:
            X_train_scaled[nfeat] = std_scaler.fit_transform(np.array(X_train[nfeat]).reshape(-1, 1)).squeeze()
            X_test_scaled[nfeat] = std_scaler.transform(np.array(X_test[nfeat]).reshape(-1, 1)).squeeze()

    return X_train_scaled, X_test_scaled
