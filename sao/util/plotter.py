import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pathlib


class Plot:
    """
    This class is used to generate the real-time convergence plots of an optimization loop.
    """

    def __init__(self, names, path=None, **kwargs):
        """

        :param names: A list of strings for the names of the responses you want to plot (i.e. objective, constr_1, etc.)
        :param path: The path were you want these plots to be saved. If no path is given, the plots are not saved.
        :param kwargs:
        """
        self.names = names
        plt.ion()
        self.fig = []
        for i in range(0, len(self.names)):
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
        self.path = pathlib.Path(path) if path is not None else None
        self.iter = 0

    def plot(self, y_values, **kwargs):
        """
        This function plots the convergence of whatever you give it as the optimization runs.

        :param y_values: The response function values you want to plot.
        :param kwargs:
        :return:
        """
        for idx, yi in enumerate(y_values):
            ax = self.fig[idx].gca()
            self.lines[idx].set_xdata(np.append(self.lines[idx].get_xdata(), np.ones_like(yi)*self.iter))
            self.lines[idx].set_ydata(np.append(self.lines[idx].get_ydata(), yi))
            ax.relim()
            ax.autoscale_view()
            self.fig[idx].canvas.draw()
            self.fig[idx].canvas.flush_events()
        self.iter += 1
        if self.path is not None:
            self.save_fig()
        plt.show()

    def save_fig(self):
        for idx, fig in enumerate(self.fig):
            plt.figure(fig.number)
            path = self.path.joinpath(f'{self.names[idx]}.png')
            plt.savefig(path)


class Plot2:
    """
    This class is used to generate the pair plots of specific {g_j - x_i}. On the same graph, exact plots of
    {g_j - x_i} and approximate plots of {g_j_tilde - x_i} are shown in order to asses the quality of the generated
    approximation. Since all approximations must be separable (for the solver to be efficient), the exact plots show
    the best function approximation one can achieve with such (separable) analytical Taylor-like expansions. Also,
    by studying such plots, one can get a better understanding as to what intervening variables would be beneficial.
    """
    def __init__(self, prob, **kwargs):
        """

        :param prob: From the prob object you need to extract the bounds for the variables.
        :param kwargs: From the kwargs you get which pairs of {g_j - x_i} you want to plot. If none are given, all
                       possible combinations are plotted, that is n*(m+1) plots.
        """
        self.iter = 0
        self.x = np.linspace(prob.xmin, prob.xmax, 50).T
        self.resp = kwargs.get('responses', np.arange(0, prob.m + 1))
        self.vars = kwargs.get('variables', np.arange(0, prob.n))
        self.fig = []
        self.fig_idx = {}
        for i in self.vars:
            for j in self.resp:
                self.fig.append(plt.subplots(1, 1)[0])
                self.fig_idx[j, i] = plt.gcf().number

    def plot_pair(self, x_k, f, prob, subprob, itte):
        """
        This function plots (some of) the {g_j - x_i}, for j in responses & i in variables.

        :param x_k: The current design.
        :param f: The current response values.
        :param prob: This object is used to evaluate the exact responses, i.e. prob.g.
        :param subprob: This object is used to get several useful data for the plots, e.g. n, m, g, inter, etc.
        :param itte: Current iteration number to show on the title of the plots.
        :return:
        """

        # Initialize plotting arrays for g_j(x_curr) and g_j_tilde(x_curr)
        prob_response_array = np.empty([subprob.m + 1, self.x.shape[1]])
        approx_response_array = np.empty([subprob.m + 1, self.x.shape[1]])

        # For all design vars x_i
        for i in self.vars:

            # Make x_curr = x_k so that all design vars remain equal to x_k apart from the one you sweep
            x_curr = x_k.copy()

            # Sweep design variable x_curr[i, 1], while keeping all others ejual to the approx.x_k
            for k in range(0, self.x.shape[1]):

                # Update current point x_curr
                x_curr[i] = self.x[i, k].copy()

                # Run problem at x_curr for plotting purposes, i.e. calculate g_j(x_curr)
                prob_response_array[:, k] = prob.g(x_curr)

                # Store g_j_tilde(x_curr) to an array for plotting purposes
                approx_response_array[:, k] = subprob.g(x_curr)

            # For all responses g_j
            for j in self.resp:

                # Get current figure and axes handles
                plt.figure(self.fig_idx[j, i])
                ax = plt.gca()

                # Get maximum values for y-axis so it keeps a nice scale
                if self.iter > 0:
                    y_min = min(min(prob_response_array[j, :]), ax.get_ylim()[0])
                    y_max = max(max(prob_response_array[j, :]), ax.get_ylim()[1])
                else:
                    y_min = min(prob_response_array[j, :])
                    y_max = max(prob_response_array[j, :])

                # Plot new exact response
                if self.iter % 2 == 1:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'b',
                                          label='$g_{}$'.format({j}) + '$^{(}$' +
                                                '$^{}$'.format({itte}) + '$^{)}$')
                else:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'r',
                                          label='$g_{}$'.format({j}) + '$^{(}$' +
                                                '$^{}$'.format({itte}) + '$^{)}$')

                # Plot asymptotes (commented out) and force to NaN values farther than asymptotes for MMA_based
                if subprob.inter.__class__.__name__ == 'MMA':
                    # L_i = plt.axvline(x=approx.low[i], color='g', label='$L_{}^{{(k)}}$'.format(i + 1))
                    # U_i = plt.axvline(x=approx.upp[i], color='y', label='$U_{}^{{(k)}}$'.format(i + 1))

                    # Put = NaN the points of g_j_tilde that x_i > U_i and x_i < L_i
                    for k in range(0, self.x.shape[1]):
                        if (self.x[i, k] <= 1.01 * subprob.inter.low[i]) or (self.x[i, k] >= 0.99 * subprob.inter.upp[i]):
                            approx_response_array[j, k] = np.NaN

                # Alternate between red and blue plots to tell them apart easily
                if self.iter % 2 == 1:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'b--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({itte}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i + 1}) +
                                               '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='o', markersize=9)
                    alpha = plt.axvline(x=subprob.ml.alpha[i], color='b', linestyle=(0, (3, 8)),
                                        label=fr'$\alpha_{i}^{{({itte})}}$')
                    beta = plt.axvline(x=subprob.ml.beta[i], color='b', linestyle=(0, (3, 8, 1, 8)),
                                       label=fr'$\beta_{i}^{{({itte})}}$')
                else:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'r--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({itte}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i + 1}) +
                                               '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='s', markersize=9)
                    alpha = plt.axvline(x=subprob.ml.alpha[i], color='r', linestyle=(0, (3, 8)),
                                        label=fr'$\alpha_{i}^{{({itte})}}$')
                    beta = plt.axvline(x=subprob.ml.beta[i], color='r', linestyle=(0, (3, 8, 1, 8)),
                                       label=fr'$\beta_{i}^{{({itte})}}$')

                # Delete the plot for (k-2), i.e. L_ji, U_ji, g_i(X), g_i_tilde(X) & respective legends
                if self.iter > 1:
                    for m in range(0, 5):  # Change 3 to 5 if you wanna plot asymptotes cuz you add 2 more lines
                        ax.lines[0].remove()

                # Plot details (labels, title, x_limit, y_limit, fontsize, legend, etc)
                x_min = self.x[i, 0].copy()
                x_max = self.x[i, -1].copy()

                # Print approximation name for each pair of {x_i, g_j} and set limits for x and y axes
                ax.set(xlabel=f'$x_{i}$', ylabel=f'$g_{j}$',
                       xlim=(x_min - 0.01 * (x_max - x_min), x_max + 0.01 * (x_max - x_min)),
                       ylim=(y_min - 0.01 * (y_max - y_min), y_max + 0.01 * (y_max - y_min)),
                       title='%s: {},  $iter = {}$'.format(subprob.approx.__class__.__name__, itte) % prob.__class__.__name__)

                # FontSize for title, xlabel and ylabel set to 20
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                    item.set_fontsize(20)
                plt.grid(True)
                plt.legend(loc='upper right')
                plt.show(block=False)

        self.iter += 1
