import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class Plot:
    def __init__(self, names, **kwargs):
        self.names = names
        plt.ion()
        self.fig = []
        for i, s in enumerate(self.names):
            self.fig.append(plt.subplots(1, 1)[0])
        self.nplot = len(self.names)
        self.cols = [colorsys.hsv_to_rgb(1.0 / (self.nplot + 1) * v, 1.0, 1.0) for v in np.arange(0, self.nplot)]
        self.lines = [None] * self.nplot
        for idx, name in enumerate(self.names):
            ax = self.fig[idx].gca()
            ax.set_title(name)
            ax.set_xlabel("iterations")
            ax.set_ylabel("value")
            ax.grid(True)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            self.lines[idx], = ax.plot([], [], marker='.', color=self.cols[idx], label=name)
        self.iter = 0

    def live_plot(self, y_values, **kwargs):
        for idx, yi in enumerate(y_values):
            ax = self.fig[idx].gca()
            self.lines[idx].set_xdata(np.append(self.lines[idx].get_xdata(), np.ones_like(yi)*self.iter))
            self.lines[idx].set_ydata(np.append(self.lines[idx].get_ydata(), yi))
            ax.relim()
            ax.autoscale_view()
            self.fig[idx].canvas.draw()
            self.fig[idx].canvas.flush_events()
        self.iter += 1
        plt.show()
