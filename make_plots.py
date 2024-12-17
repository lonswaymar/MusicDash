import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np

COLOR_LIST = ["#145A32", "#97bc62"]

def make_barplot(highlight_genres, data):
    '''doc string'''
    
    global COLOR_LIST
    
    fig, ax = plt.subplots(figsize=(8,5))
    
    n = 7 # number of genres
    sorted_genres = data["UpdatedGenre"].value_counts()[:n].sort_values(ascending=True)
    
    # Make a list of top n lowercase genres without spaces, hyphens. 
    genre_list = [genre for genre in sorted_genres.index]  
    
    colors = []
    j = 0
    for genre in genre_list:
        if genre in highlight_genres:
            colors.append(COLOR_LIST[j])
            j += 1
        else:
            colors.append("grey")
        
    
    sorted_genres.plot(kind='barh', ax=ax, color=colors)
    
    # Adjust the spines
    ax.spines['right'].set_visible(False)  # Remove the right vertical line
    ax.spines['top'].set_visible(False)    # Remove the top horizontal line
    ax.spines['left'].set_visible(False)  
    ax.spines['bottom'].set_visible(False) 
    
    # adjust axis labels -- show nothing for this chart
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis='x', length=0) 
    ax.tick_params(axis='y', length=0, labelsize=11) 
    ax.set_ylabel("")
    
    # Titles
    plt.text(0, 1.052, "Quantity of Genres from given Spotify Playlist", transform=ax.transAxes, fontsize=22, color='k', va='bottom', ha='left') 
    plt.text(0, 1.0, "Genres were simplified from original Spotify descriptions", transform=ax.transAxes, fontsize=14, color='grey', va='bottom', ha='left') 

    
    # Add labels to end of each bar
    for p, color in zip(ax.patches, colors):
        s = f"{p.get_width():d}"
        ax.text(p.get_width()+0.5, p.xy[1]+0.12, s, color=color, fontsize = 13)
    
    plt.savefig("./static/images/barchart.png")
    
    

def make_boxplot(genres, df):
    
    global COLOR_LIST
    
    fig, ax = plt.subplots(figsize=(9,6))


    filtered = df[df['UpdatedGenre'].isin(genres)]
        
    ordered_genres = filtered["UpdatedGenre"].value_counts().sort_values(ascending=True).index.to_list()
    
    if ordered_genres[0] > ordered_genres[1]:
        colors = [COLOR_LIST[1], COLOR_LIST[0]]
    else: 
        colors = COLOR_LIST
    

    filtered.boxplot(column='Duration', by='UpdatedGenre', ax=ax, patch_artist=True, color="k")

    # Set colors for each box
    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color)
    
    # Adjust the spines
    ax.spines['right'].set_visible(False)  # Remove the right vertical line
    ax.spines['top'].set_visible(False)    # Remove the top horizontal line

    ax.tick_params(axis='x', length=0, labelsize=16) 
    ax.tick_params(axis='y', length=0, labelsize=16) 
    ax.grid(alpha=0)
    
    
    ax.set_xlabel("")
    ax.set_title("")
    fig.suptitle("")
    
    plt.text(0, 1.062, "Duration of Songs in Each Genre", transform=ax.transAxes, fontsize=24, color='k', va='bottom', ha='left') 
    plt.text(0, 1.0, "Outliers May Break Assumptions of Normality", transform=ax.transAxes, fontsize=14, color='grey', va='bottom', ha='left') 
    
        
    plt.savefig("./static/images/boxplot.png")
    
    return(ordered_genres, colors)
        

    
def make_distribution(genres, df, colors):
    
    fig, ax = plt.subplots(figsize=(9,6))
    
    genre1Data = df[df["UpdatedGenre"] == genres[0]].Duration
    genre2Data = df[df["UpdatedGenre"] == genres[1]].Duration
    
    ax.hist(genre1Data, bins=20, color=colors[0], alpha=0.6, histtype="step", linewidth=5)
    ax.hist(genre2Data, bins=20, color=colors[1], alpha=0.6, histtype="step", linewidth=5)
    
    # Adjust the spines
    ax.spines['right'].set_visible(False)  # Remove the right vertical line
    ax.spines['top'].set_visible(False)    # Remove the top horizontal line
    ax.tick_params(axis='x', length=0, labelsize=14) 
    ax.tick_params(axis='y', length=0, labelsize=14) 
    
    ax.xaxis.label.set_fontsize(18)  
    ax.yaxis.label.set_fontsize(18)
    
    ax.set_xlabel("Time (seconds)", fontsize=20)  
    
    plt.text(0, 1.067, "Are Distributions Normal?", transform=ax.transAxes, fontsize=22, color='k', va='bottom', ha='left') 
    plt.text(0, 1.01, "Variance of each populated calculated to discern test type", transform=ax.transAxes, fontsize=14, color='grey', va='bottom', ha='left') 
    
    # calculate t statistics
    genre1Var = np.var(genre1Data)
    genre2Var = np.var(genre2Data)
    
    variances = [genre1Var, genre2Var]
    
    indexLargestVariance = np.argmax(variances)
    indexSmallerVariance = np.argmin(variances)
    
    largerVariance = variances[indexLargestVariance]
    smallerVariance = variances[indexSmallerVariance]
    
    if largerVariance <= 4*smallerVariance:
        equal_var = True
        dof = len(genre1Data) + len(genre2Data) - 2
    else:
        equal_var = False
        # Variance calculations
        s1 = genre1Var / len(genre1Data)  # Variance contribution of genre1
        s2 = genre2Var / len(genre2Data)  # Variance contribution of genre2
        
        # Welch-Satterthwaite Degrees of Freedom
        numerator = (s1 + s2) ** 2
        denominator = (s1 ** 2 / (len(genre1Data) - 1)) + (s2 ** 2 / (len(genre2Data) - 1))
        dof = numerator / denominator
    
    results = stats.ttest_ind(a=genre1Data, b=genre2Data, equal_var=equal_var)
    
    
    plt.savefig("./static/images/distributions.png")
    
    return results, dof
    
    
def make_tdistribution(test_results, dof):
    statistic, pvalue = test_results
    
    fig, ax = plt.subplots(figsize=(8,4))
    
    leftMost = -4
    rightMost = 4
    if ((test_results[0] < leftMost) | (test_results[0] > rightMost)):
        leftMost = -np.abs(test_results[0])
        rightMOst = np.abs(test_results[0])
        
    xvals = np.linspace(leftMost, rightMost,1000)
    
    pdf = stats.t.pdf(xvals, df=dof)
    
    alpha=0.05
    t_critical_positive = stats.t.ppf(1 - alpha / 2, df=dof)  # Upper tail
    t_critical_negative = stats.t.ppf(alpha / 2, df=dof)  # Lower tail
    
    
    
    ax.plot(xvals, pdf, 'k-', linewidth=3)
    
    x_shade = np.linspace(np.min(xvals), t_critical_negative, 50)
    y_shade = stats.t.pdf(x_shade, df=dof)
    ax.fill_between(x_shade, y_shade, color='black', alpha=0.5)
    
    x_shade = np.linspace(t_critical_positive, np.max(xvals), 50)
    y_shade = stats.t.pdf(x_shade, df=dof)
    ax.fill_between(x_shade, y_shade, color='black', alpha=0.5)

    ax.vlines(test_results[0], 0, np.max(pdf), color="navy") 
    ax.vlines(0, 0, np.max(pdf), linestyle="dashed", alpha=0.4, color="grey")
    
    
    # Adjust the spines
    ax.spines['right'].set_visible(False)  # Remove the right vertical line
    ax.spines['top'].set_visible(False)    # Remove the top horizontal line
    ax.spines['left'].set_visible(False)    # Remove the top horizontal line
    ax.tick_params(axis='x', length=0, labelsize=14) 
    ax.tick_params(axis='y', length=0, labelsize=14)    
    ax.set_yticklabels([])
    ax.set_xticks([0])      # the positions I want the labels
    ax.set_xticklabels(["t = 0"], fontsize=16, color=(0,0,0,.75)) 
    
    plt.text(0, 1.067, "Location of Test Statistic on the t-distribution", transform=ax.transAxes, fontsize=18, color='k', va='bottom', ha='left') 
    plt.text(0, 1.01, f"P Value = {test_results[1]:.2e}", transform=ax.transAxes, fontsize=14, color='grey', va='bottom', ha='left') 
    
    
    plt.savefig("./static/images/statistic.png")
    
    
    
    
