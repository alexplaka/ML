import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def draw_plot():
    
    df = pd.read_csv("epa-sea-level.csv")  # Read data from file

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["Year"], df["CSIRO Adjusted Sea Level"], 
               marker='.', color='c', label="Data")

    # Create first line of best fit
    lreg1 = linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    x1 = np.array(list(range(1880, 2051)))
    ax.plot(x1, lreg1.intercept + lreg1.slope * x1, 'g',
            label="Linear trend from $1880$-$2013$")
    print(f"\nLinear regression from 1880-2013:\n{lreg1}")

    # Create second line of best fit
    lreg2 = linregress(df[df["Year"] >= 2000]["Year"], 
                       df[df["Year"] >= 2000]["CSIRO Adjusted Sea Level"])
    x2 = np.array(list(range(2000, 2051)))
    ax.plot(x2, lreg2.intercept + lreg2.slope * x2, 'r', 
            label="Linear trend from $2000$-$2013$")
    print(f"\nLinear regression from 2000-2013:\n{lreg2}")
    
    # Add labels and title
    ax.set(title="Rise in Sea Level (Extrapolated to $2050$)", 
           xlabel="Year", 
           ylabel="Sea Level (inches)",
           xlim=(1875, 2060))
    
    ax.legend()
    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    
    return ax

if __name__ == "__main__":
    draw_plot()