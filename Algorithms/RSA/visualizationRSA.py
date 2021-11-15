import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 

def plotListenerActionDistribution(df, yAxisLabel  = 'Probability', save=False, filename = 'listenerDistribution', figSize = (7, 3), barColorHex = '#47802b'):
    sns.set(style="white", context="talk")

    f, ax1 = plt.subplots(1, 1, figsize=figSize, sharex=True)

    columnName = df.columns[0]
    x = df.index
    y1 = df[columnName].values
    sns.barplot(x=x, y=y1, color = barColorHex, ax=ax1)

    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel(yAxisLabel)
    ax1.set_xticklabels(x, rotation=30)

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)
    if save:
        plt.savefig("S_{}.png".format(filename))



def plotSignalerActionDistribution(signalerDataframe, yAxisLabel  = 'Probability', save=False, filename = 'signalerDistribution', figSize = (7, 3), barColorHex = '#045658'):
    sns.set(style="white", context="talk")

    # Set up the matplotlib figure
    f, ax1 = plt.subplots(1, 1, figsize=figSize, sharex=True)

    columnName = signalerDataframe.columns[0]
    # Generate some sequential data
    x = signalerDataframe.index.values
    y1 = signalerDataframe[columnName].values
    sns.barplot(x=x, y=y1, color = barColorHex, ax=ax1)

    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel(yAxisLabel)

    ax1.set_xticklabels(x, rotation=30)

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=2)
    if save:
        plt.savefig("S_{}.png".format(filename))