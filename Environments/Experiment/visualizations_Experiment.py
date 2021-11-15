import sys
import numpy as np
import pandas as pd
import itertools

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, RegularPolygon, Circle
import matplotlib as mpl



def plotSignalerActionDistribution(signalerDataframe, save = False, filename = 'signalDistribution', figuresize = (7, 3), color = '#045658', ylabel = 'probability'):
    sns.set(style="white", context="talk")

    # Set up the matplotlib figure
    f, ax1 = plt.subplots(1, 1, figsize=figuresize, sharex=True)

    # extract correct x, y from pandas dataframe
    x = signalerDataframe.index.get_level_values('signals')
    y1 = signalerDataframe['probability'].values
    sns.barplot(x=x, y=y1, color = color, ax=ax1)

    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel(ylabel)
    ax1.set_xticklabels(x, rotation=30)
    ax1.set_ylim(0, 1)
    

    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[0,.5,1])
    plt.tight_layout(h_pad=2)
    if save:
    	plt.savefig("Sig_{}.png".format(filename))



def plotReceiverActionDistributions(naiveReceiverDataframe, pragmaticReceiverDataframe,locationDictionary, receiverLocInaction, signal, save = False, filename = 'actionDistribution', figuresize = (7, 5), color = '#47802b', ylabel = 'probability'):
    naiveReceiverActionProbabilities = sumAcrossActions(naiveReceiverDataframe)
    pragmaticReceiverActionProbabilities = sumAcrossActions(pragmaticReceiverDataframe)
    
    naiveActions = mapLocationsToObjects(naiveReceiverActionProbabilities, locationDictionary, receiverLocInaction)
    pragmaticActions = mapLocationsToObjects(pragmaticReceiverActionProbabilities, locationDictionary, receiverLocInaction)
    
    sns.set(style="white", context="talk")

    # Set up the matplotlib figure
    f, (ax1, ax2) = plt.subplots(2, 1, figsize=figuresize, sharex=True)

    x = naiveActions
    
    y1 = naiveReceiverActionProbabilities.values
    sns.barplot(x=x, y=y1, color = color, ax=ax1)
    
    y2 = pragmaticReceiverActionProbabilities.values
    sns.barplot(x=x, y=y2, color = color, ax=ax2)

    ax1.axhline(0, color="k", clip_on=False)
    ax1.set_ylabel("Naive")
    ax1.set_ylim(0, 1)
    
    ax2.axhline(0, color="k", clip_on=False)
    ax2.set_ylabel("Pragmatic")
    ax2.set_ylim(0, 1)

    ax2.set_xticklabels(x, rotation=30)
    ax1.set_title("Signal: {}".format(signal))
    # Finalize the plot
    sns.despine(bottom=True)
    plt.setp(f.axes, yticks=[0,.5,1])
    plt.tight_layout(h_pad=2)

    if save:
    	f.savefig("Rec_{}.png".format(filename))


def sumAcrossActions(originalReceiverDF):
    getReceiverAction = lambda x: x.index.get_level_values('actions')[0][1]
    originalReceiverDF['receiver'] = originalReceiverDF.groupby(originalReceiverDF.index.names).apply(getReceiverAction)
    receiverActionSeries = originalReceiverDF.groupby(by=["receiver"])["p(mind|signal)"].sum()
    return(receiverActionSeries)

def mapLocationsToObjects(receiverActionSeries, locationDictionary, receiverLocInaction):
    listOfLocations = receiverActionSeries.index.get_level_values('receiver')
    targetList = ['stay' if location == receiverLocInaction else locationDictionary[location] for location in listOfLocations]
    return(targetList)



def visualizeExperimentTrial(widthHeightTuple, signalerLocation, receiverLocation, signalStates = {}, possibleTargets = {}, barriers = None, signalsAsVocab = True, save=False, filename = './exptTrial.plotSignalerActionDistribution'):
    states = set(itertools.product(range(widthHeightTuple[0]), range(widthHeightTuple[1])))
    gridAdjust = .5
    gridScale = .7

    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]

    #fig = plt.figure(figsize =((maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale))
    #ax = fig.add_axes([0,0,1,1])
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon = False, xticks = range(minimumx - 1, maximumx + 2), yticks = range(minimumy - 1, maximumy + 2))

    # gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    # Signaler and Receiver drawing
    ax.text(signalerLocation[0]-.1, signalerLocation[1]-.1, "S", fontsize = 20)
    ax.text(receiverLocation[0]-.1, receiverLocation[1]-.1, "R", fontsize = 20)
    
    #Signal coloring and labeling
    if signalsAsVocab:
        listString = ["Vocab: \n "] + [", ".join(signalStates)]
        signalString = " ".join(listString)
        ax.text(maximumx+1.0, (maximumy-minimumy)/2, signalString, fontsize =15)
    else:
        for (signalx,signaly), signal in signalStates.items():
            ax.add_patch(Rectangle((signalx-gridAdjust, signaly-gridAdjust), 1, 1, fill=False, edgecolor='white', linewidth = 4))
           
            if 'green' in signal:
                ax.add_patch(Rectangle((signalx-gridAdjust, signaly-gridAdjust), 1, 1, fill=True, color='#56bc52'))
            if 'blue' in signal:
                ax.add_patch(Rectangle((signalx-gridAdjust, signaly-gridAdjust), 1, 1, fill=True, color='blue'))
            if 'purple' in signal:
                ax.add_patch(Rectangle((signalx-gridAdjust, signaly-gridAdjust), 1, 1, fill=True, color='#906ae7'))
            if 'red' in signal:
                ax.add_patch(Rectangle((signalx-gridAdjust, signaly-gridAdjust), 1, 1, fill=True, color='#d40a00'))
            if 'circle' in signal:
                ax.add_patch(Circle((signalx, signaly),.3, fill=False, color='black'))
            if 'triangle' in signal:
                ax.add_patch(RegularPolygon((signalx, signaly),3,.3, fill=False, color='black'))
            if 'square' in signal:
                ax.add_patch(Rectangle((signalx-gridAdjust/2, signaly-gridAdjust/2),.5,.5,  fill=False, color='black', linewidth=2))
        
        
    #Target Item coloring and labeling
    for (targetx, targety), targetFeatures in possibleTargets.items():
        colorDict = {'green': '#56bc52', 'purple': '#906ae7', 'red': '#d40a00', 'grey': '#808080'}
        targetColor = targetFeatures.split()[0]
        
        if 'circle' in targetFeatures:
            ax.add_patch(Circle((targetx, targety),.3, fill=True, color = colorDict[targetColor], linewidth = 2))
        if 'triangle' in targetFeatures:
            ax.add_patch(RegularPolygon((targetx, targety),3,.3, fill=True, color = colorDict[targetColor], linewidth=2))
        if 'square' in targetFeatures:
            ax.add_patch(Rectangle((targetx-gridAdjust/2, targety-gridAdjust/2),.5,.5, fill=True, color = colorDict[targetColor], linewidth=2))

    if barriers is not None:
        for (state, action) in barriers:

            ax.add_line(mpl.lines.Line2D([state[0], state[0]+action[0]], [state[1], state[1]+action[1]], color = 'firebrick'))

    if save:
        plt.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()


def visualizeExperimentTrial_Tight(widthHeightTuple, signalerLocation, receiverLocation, possibleTargets = {}, save=False, filename = './exptTrial'):
    states = set(itertools.product(range(widthHeightTuple[0]), range(widthHeightTuple[1])))
    gridAdjust = .5
    gridScale = .7

    minimumx, minimumy = [min(coord) for coord in zip(*states)]
    maximumx, maximumy = [max(coord) for coord in zip(*states)]

    #fig = plt.figure(figsize =((maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale))
    #ax = fig.add_axes([0,0,1,1])
    plt.rcParams["figure.figsize"] = [(maximumx-minimumx)*gridScale, (maximumy-minimumy)*gridScale]
    ax = plt.gca(frameon = False, xticks = range(minimumx - 1, maximumx + 2), yticks = range(minimumy - 1, maximumy + 2))

    # gridline drawing
    for (statex, statey) in states:
        ax.add_patch(Rectangle((statex-gridAdjust, statey-gridAdjust), 1, 1, fill=False, color='black', alpha=1))

    # Signaler and Receiver drawing
    ax.text(signalerLocation[0]-.1, signalerLocation[1]-.1, "S", fontsize = 20)
    ax.text(receiverLocation[0]-.1, receiverLocation[1]-.1, "R", fontsize = 20)

        
    #Target Item coloring and labeling
    for (targetx, targety), targetFeatures in possibleTargets.items():
        colorDict = {'green': '#56bc52', 'purple': '#906ae7', 'red': '#d40a00', 'grey': '#808080'}
        targetColor = targetFeatures.split()[0]
        
        if 'circle' in targetFeatures:
            ax.add_patch(Circle((targetx, targety),.3, fill=True, color = colorDict[targetColor], linewidth = 2))
        if 'triangle' in targetFeatures:
            ax.add_patch(RegularPolygon((targetx, targety),3,.3, fill=True, color = colorDict[targetColor], linewidth=2))
        if 'square' in targetFeatures:
            ax.add_patch(Rectangle((targetx-gridAdjust/2, targety-gridAdjust/2),.5,.5, fill=True, color = colorDict[targetColor], linewidth=2))

    plt.tick_params(axis='both',which='both',bottom=False,left=False,labelbottom=False, labelleft=False)

    if save:
        plt.savefig(filename,dpi=300, bbox_inches='tight')
    plt.show()