import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("website-pageviews.csv", index_col="date", parse_dates=True)

df = df[df.value.between(df.value.quantile(0.025), df.value.quantile(0.975))].copy()


def draw_line_plot():
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(df, 'r')
    ax.set(title="Daily Page Views 5/2016-12/2019",
           xlabel="Date",
           ylabel="Page Views")

    fig.savefig('line_plot.png')

    return fig


def draw_bar_plot():
    df_bar = df.copy()
    df_bar["Month"] = df_bar.index.month_name()
    df_bar["Year"] = df_bar.index.year

    df_bar_grouped = df_bar.groupby(["Year", "Month"]).mean()
    df_bar_grouped_ind = df_bar_grouped.index.to_frame()

    fig = sns.barplot(x=df_bar_grouped_ind.Year, y="value",
                      hue=df_bar_grouped_ind.Month,
                      data=df_bar_grouped,
                      hue_order=['January', 'February', 'March', 'April', 'May',
                                 'June', 'July', 'August', 'September', 'October',
                                 'November', 'December'])
    fig.axes.set_ylabel('Average Page Views')
    fig.figure.set_figwidth(7)
    fig.figure.set_figheight(7)
    fig.axes.legend(title=None, loc='upper left')
    fig.axes.set_title(None)

    fig.figure.savefig('bar_plot.png')

    return fig


def draw_box_plot():
    df_box = df.copy()
    df_box["Month"] = df_box.index.month_name()
    df_box["Year"] = df_box.index.year

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(x="Year", y="value", data=df_box, ax=ax[0])
    ax[0].set(title="Year-wise Box Plot (Trend)",
              ylabel="Page Views")

    sns.violinplot(x="Month", y="value", data=df_box, ax=ax[1],
                   order=['January', 'February', 'March', 'April', 'May',
                          'June', 'July', 'August', 'September', 'October',
                          'November', 'December'])
    ax[1].set(title="Month-wise Violin Plot (Seasonality)",
              ylabel="Page Views",
              xticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May',
                           'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                           'Nov', 'Dec'])

    fig.savefig('box_plot.png')

    return fig


if __name__ == "__main__":
    draw_line_plot()
    draw_bar_plot()
    draw_box_plot()
