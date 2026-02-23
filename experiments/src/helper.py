import matplotlib.pyplot as plt
import numpy as np



def plot_predictions_mask(dates_grid, index, y, y_pred, train_start_date, test_start_date):

    fig, ax = plt.subplots()
    ax.plot(dates_grid[index >= train_start_date], y)
    ax.plot(dates_grid[index >= train_start_date], y_pred, "--")
    ax.vlines(x=test_start_date, ymin=0.95 * np.min(y), ymax=1.05 * np.max(y), color="k")