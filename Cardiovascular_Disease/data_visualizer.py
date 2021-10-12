import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


def draw_cat_plot(df, id_var, cat_feats, *, output_filename=None):
    """ Draw plot showing value counts of categorical features. """

    # Create DataFrame for cat plot using `pd.melt` using just the values from categorical features
    df_cat = pd.melt(df, id_vars=id_var, value_vars=cat_feats)

    # Draw the catplot
    fig = sns.catplot(x="variable", hue="value", col=id_var, data=df_cat, 
                      kind="count")
    fig.set_xlabels('')
    fig.set_xticklabels(rotation=90)
    fig.savefig(output_filename) if output_filename is not None else True

    return fig


def draw_corr_matrix(df):
    """
    - Draw correlation matrix as heatmap.
    - Draw correlation for target feature and mutual information in a bar plot.
    """

    corr = df.corr()  # Calculate the correlation matrix
    cardio_corr = corr.loc["Cardio", :"Active"]  # Correlation for the target
    mi = mutual_info_classif(df.iloc[:, :-1], df["Cardio"])  # Calculate MI score
    scores = cardio_corr.to_frame()
    scores.rename(columns={"Cardio": "Corr"}, inplace=True)
    scores["MI"] = mi
    scores_melted = pd.melt(scores, ignore_index=False)

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True  # Generate a mask for the upper triangle

    fig, ax = plt.subplots(2, 1, figsize=(8, 15), dpi=100)
    sns.heatmap(corr, mask=mask, square=True, ax=ax[0])
    ax[0].set_title("Feature Correlation", fontdict={"fontsize": 14})

    # Plot the "Cardio" correlation and mutual informatuion scores on the sme graph.
    sns.barplot(x="value", y=scores_melted.index, hue="variable", 
                data=scores_melted, ax=ax[1])
    # sns.barplot(x=[np.array(cardio_corr), mi], y=cardio_corr.index, ax=ax[1],
    #             color=[0.30, 0.41, 0.29])  # to plot just the "Cardio" correlation scores
    ax[1].set_title("Target (Cardio) Correlation and Mutual Information",
                    fontdict={"fontsize": 14})
    ax[1].set_xlabel(None)
    ax[1].legend(title=None)

    fig.savefig('Corr_matrix_Target.png')

    return fig, corr, scores
