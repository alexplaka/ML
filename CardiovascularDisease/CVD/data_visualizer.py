import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


def draw_cat_plot(df: pd.DataFrame, id_var: str, cat_feats: list, *, output_filename: str =None):
    """
    Draw plot showing value counts of categorical features.

    :parameter dframe: pandas dataframe containing the feature `id_var` and all of the features in `cat_feats`.
                       Note: this implementation does not check that all of the relevant features are in `dframe`.
    :parameter id_var: Feature name (string) with respect to which panels of the categorical plot
                       are made. For instance, for a binary feature, the plot will
                       have two panels showing the respective counts of categorical features.
    :parameter cat_feats: list of strings of categorical features to plot.
    :parameter output_filename: if the plot is to be saved, this is its name.
                                (default=None, i.e., plot is not saved)

    :return: Seaborn figure object.
    """

    # Create DataFrame for cat plot using `pd.melt` using just the values from categorical features
    df_cat = pd.melt(df, id_vars=id_var, value_vars=cat_feats)

    # Draw the catplot
    fig = sns.catplot(x="variable", hue="value", col=id_var, data=df_cat, 
                      kind="count")
    fig.set_xlabels('')
    fig.set_xticklabels(rotation=90)
    fig.savefig(output_filename) if output_filename is not None else True

    return fig


def draw_corr_matrix(df: pd.DataFrame):
    """
    - Draw correlation matrix as heatmap.
    - Draw correlation for target feature and mutual information in a bar plot.
    Note: Assuming the target feature is in the last column of df.

    :parameter df: pandas dataframe with all of the relevant features as columns.
    :return: fig: matplotlib figure object;
             corr: correlation matrix for all features;
             scores: pandas dataframe with the correlation and mutual information scores
                     for the target feature.
    """

    target = df.columns[-1]

    corr = df.corr()  # Calculate the correlation matrix
    target_corr = corr.loc[target, corr.columns.delete(-1)]  # Correlation for the target
    mi = mutual_info_classif(df.iloc[:, :-1], df[target])  # Calculate MI score
    scores = target_corr.to_frame()
    scores.rename(columns={target: "Corr"}, inplace=True)
    scores["MI"] = mi
    scores_melted = pd.melt(scores, ignore_index=False)

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True  # Generate a mask for the upper triangle

    fig, ax = plt.subplots(2, 1, figsize=(8, 15), dpi=100)
    sns.heatmap(corr, mask=mask, square=True, ax=ax[0], cmap='Spectral_r',
                annot=True, fmt='.2f', annot_kws={'fontsize': 8})
    ax[0].set_title("Feature Correlation", fontdict={"fontsize": 14})

    # Plot the "Cardio" correlation and mutual information scores on the sme graph.
    sns.barplot(x="value", y=scores_melted.index, hue="variable", 
                data=scores_melted, ax=ax[1], palette='crest')
    # sns.barplot(x=[np.array(cardio_corr), mi], y=cardio_corr.index, ax=ax[1],
    #             color=[0.30, 0.41, 0.29])  # to plot just the "Cardio" correlation scores
    ax[1].set_title(f"Target ({target}) Correlation and Mutual Information",
                    fontdict={"fontsize": 14})
    ax[1].set_xlabel(None)
    ax[1].legend(title=None)
    ax[1].grid(axis='x')

    fig.savefig('Corr_matrix_Target.png')

    return fig, corr, scores
