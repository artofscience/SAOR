import numpy as np
import matplotlib.pyplot as plt


class Plot2:
    def __init__(self, prob, **kwargs):
        self.iter = 0
        self.x = np.linspace(prob.xmin, prob.xmax, 50).T

    # This function plots all g_j - x_i, for j = 0, 1, ..., m & i = 0, 1, ..., n-1
    def plot_approx(self, x_k, f, prob, subprob):

        fig_count = 0

        # Initialize plotting arrays for g_j(x_curr) and g_j_tilde(x_curr)
        prob_response_array = np.empty([subprob.m + 1, self.x.shape[1]])
        approx_response_array = np.empty([subprob.m + 1, self.x.shape[1]])

        # For all design vars x_i
        for i in range(0, subprob.n):

            # Make x_curr = x_k so that all design vars remain ejual to x_k apart from the one you sweep
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
            for j in range(0, prob.m + 1):

                # Get current figure and axes handles
                plt.figure(fig_count)
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
                                                '$^{}$'.format({self.iter}) + '$^{)}$')
                else:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'r',
                                          label='$g_{}$'.format({j}) + '$^{(}$' +
                                                '$^{}$'.format({self.iter}) + '$^{)}$')

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
                                                  '$^{}$'.format({self.iter}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i + 1}) +
                                               '$^{(}$' + '$^{}$'.format({self.iter}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='o', markersize=9)
                else:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'r--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({self.iter}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i + 1}) +
                                               '$^{(}$' + '$^{}$'.format({self.iter}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='s', markersize=9)

                # Delete the plot for (k-2), i.e. L_ji, U_ji, g_i(X), g_i_tilde(X) & respective legends
                if self.iter > 1:
                    for m in range(0, 3):  # Change 3 to 5 if you wanna plot asymptotes cuz you add 2 more lines
                        ax.lines[0].remove()

                # Plot details (labels, title, x_limit, y_limit, fontsize, legend, etc)
                x_min = self.x[i, 0].copy()
                x_max = self.x[i, -1].copy()

                # Print approximation name for each pair of {x_i, g_j} and set limits for x and y axes
                ax.set(xlabel='$x_{}$'.format({i + 1}), ylabel='$g_{}$'.format({j}),
                       xlim=(x_min - 0.01 * (x_max - x_min), x_max + 0.01 * (x_max - x_min)),
                       ylim=(y_min - 0.01 * (y_max - y_min), y_max + 0.01 * (y_max - y_min)),
                       title='%s: {},  $iter = {}$'.format(subprob.approx.__class__.__name__, self.iter) % prob.__class__.__name__)

                # FontSize for title, xlabel and ylabel set to 20
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                    item.set_fontsize(20)
                plt.grid(True)
                plt.legend(loc='upper right')
                plt.show(block=False)
                fig_count += 1

        self.iter += 1
