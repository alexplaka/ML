import pandas as pd


def calculate_demographic_data(print_data=True):
    # Dataset Source:
	# Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. 
	# Irvine, CA: University of California, School of Information and Computer Science.
    df = pd.read_csv('adult.data.csv')  # Read data from file

    # How many of each race are represented in this dataset? This should be a Pandas series with race names as the index labels.
    race_count = df["race"].value_counts()

    # What is the average age of men?
    average_age_men = round(df[df["sex"] == "Male"]["age"].mean(), 1)
    # average_age_men = df.groupby("sex")["age"].mean()[-1]  # alternate way

    # What is the percentage of people who have a Bachelor's degree?
    percentage_bachelors = round(df[df["education"] == "Bachelors"]["education"].count() / df.shape[0] * 100, 1)

    # What percentage of people with advanced education (`Bachelors`, `Masters`, or `Doctorate`) 
    # make more than 50K?
    higher_ed_salary_counts = df[(df["education"] == "Bachelors") |
                                 (df["education"]  == "Masters") | 
                                 (df["education"] == "Doctorate")]["salary"].value_counts()
    higher_education_rich = round(higher_ed_salary_counts[1] / higher_ed_salary_counts.sum() * 100, 1)
    
    # What percentage of people without advanced education make more than 50K?
    # without `Bachelors`, `Masters`, or `Doctorate`
    lower_ed_salary_counts = df[(df["education"] != "Bachelors") &
                                 (df["education"]  != "Masters") & 
                                 (df["education"] != "Doctorate")]["salary"].value_counts()
    lower_education_rich = round(lower_ed_salary_counts[1] / lower_ed_salary_counts.sum() * 100, 1)

    # What is the minimum number of hours a person works per week (hours-per-week feature)?
    min_work_hours = df["hours-per-week"].min()

    # What percentage of the people who work the minimum number of hours per week have a salary of >50K?
    num_min_workers = df[df["hours-per-week"] == min_work_hours]["salary"].value_counts()
    rich_percentage = round(num_min_workers[1] / num_min_workers.sum() * 100, 1)

    # What country has the highest percentage of people that earn >50K?
    country_counts = dict(df["native-country"].value_counts())
    gt50K_frac = dict()
    for country, count in country_counts.items():
        sals = df[df["native-country"] == country]["salary"].value_counts()
        # Two countries do not have earners with >50K in the dataset --> append 0
        sals = sals.append(pd.Series([0], index=[">50K"])) if sals.shape[0] == 1 else sals
        gt50K_frac[country] = sals[1] / count
    highest_earning_country = max(gt50K_frac, key=gt50K_frac.get)
    highest_earning_country_percentage = round(gt50K_frac[highest_earning_country] * 100, 1)

    # Identify the most popular occupation for those who earn >50K in India.
    top_IN_occupation = df[(df["native-country"] == "India") &
                           (df["salary"] == ">50K")]["occupation"].value_counts().index[0]

    if print_data:
        print(f"Number of each race:\n{race_count}") 
        print(f"Average age of men: {average_age_men}")
        print(f"Percentage with Bachelors degrees: {percentage_bachelors}%")
        print(f"Percentage with higher education that earn >50K: {higher_education_rich}%")
        print(f"Percentage without higher education that earn >50K: {lower_education_rich}%")
        print(f"Min work time: {min_work_hours} hours/week")
        print(f"Percentage of rich among those who work fewest hours: {rich_percentage}%")
        print(f"Country with highest percentage of rich: {highest_earning_country}")
        print(f"Percentage of rich people in {highest_earning_country}: "
              f"{highest_earning_country_percentage}%")
        print(f"Top occupations in India: {top_IN_occupation}")

    return {
        'race_count': race_count,
        'average_age_men': average_age_men,
        'percentage_bachelors': percentage_bachelors,
        'higher_education_rich': higher_education_rich,
        'lower_education_rich': lower_education_rich,
        'min_work_hours': min_work_hours,
        'rich_percentage': rich_percentage,
        'highest_earning_country': highest_earning_country,
        'highest_earning_country_percentage': highest_earning_country_percentage,
        'top_IN_occupation': top_IN_occupation
    }
