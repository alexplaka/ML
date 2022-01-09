# Cardiovascular Disease

Dataset analysis and building a predictive model for the presence of cardiovascular disease in a patient.

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Dataset](#dataset)
5. [Results](#results)
6. [Licensing](#licensing)

## Installation<a id="installation"></a>

This code has been written using Python 3.9, but will run without issues in versions 3.7 and 3.8. The required libraries
were installed using the  `conda` package manager in the Anaconda distribution.

## Project Motivation<a name="motivation"></a>

For this project, I used data to address the following:

1. How the dataset's categorical features are distributed given the absence or presence of cardiovascular disease in
   patients.
2. What are the most import numerical features given their distribution?
3. Are the features correlated to each other?
4. Use different metrics to determine the most important features that are related to the presence of cardiovascular
   disease.
5. Build a predictive model using the K-nearest neighbors method.

## File Descriptions <a name="files"></a>

- The code used for this work is in the Jupyter notebook `Data_Analysis.ipynb` (found in the root directory).
    * The folder `CVD` contains files with more modular and object-oriented code for performing (more or less) the
      same analysis and building ML models.
- Images generated using the code are provided as `.png` files.

## Dataset <a name="dataset"></a>

#### Source

https://www.kaggle.com/mdshamimrahman/cardio-data-set

#### Data description

The rows in the dataset correspond to patients and the columns represent information like body measurements, results
from various blood tests, and lifestyle choices. Here, we use the dataset to explore the relationship between cardiac
disease, body measurements, blood markers, and lifestyle choices.

#### File name: `cardio_data.csv`

#### Feature details

| Feature | Variable Type | Variable      | Value Type |
|:-------:|:------------:|:-------------:|:----------:|
| Age | Objective Feature | age | int (days) |
| Height | Objective Feature | height | int (cm) |
| Weight | Objective Feature | weight | float (kg) |
| Gender | Objective Feature | gender | 1: women, 2: men, 3: unknown |
| Systolic blood pressure | Examination Feature | ap_hi | int |
| Diastolic blood pressure | Examination Feature | ap_lo | int |
| Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
| Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
| Smoking | Subjective Feature | smoke | binary, 0: no, 1: yes|
| Alcohol intake | Subjective Feature | alco | binary, 0: no, 1: yes |
| Physical activity | Subjective Feature | active | binary, 0: no, 1: yes |
| Presence of cardiovascular disease | Target Variable | cardio | binary, 0: no, 1: yes |
