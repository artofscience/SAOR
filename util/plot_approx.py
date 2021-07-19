import numpy as np
import logging
from Problems.Polynomial_1D import Polynomial1D
import sao
import matplotlib.pyplot as plt


# Set options for logging data: https://www.youtube.com/watch?v=jxmzY9soFXg&ab_channel=CoreySchafer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

# If you want to print on terminal
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


class PlotApprox:
    def __init__(self, prob, **kwargs):
        self.iter_pair = 0
        self.x = np.linspace(prob.xmin, prob.xmax, 50).T
        self.resp = kwargs.get('responses', np.arange(0, prob.m + 1))
        self.vars = kwargs.get('variables', np.arange(0, prob.n))
        self.fig = []
        self.fig_idx = {}

    def plot_approximation(self, x_k, f, prob, subprob):
        # Create a list of figures to plot on
        if self.iter_pair == 0:
            for i in self.vars:
                for j in self.resp:
                    self.fig.append(plt.subplots(1, 1)[0])
                    self.fig_idx[j, i] = plt.gcf().number

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
                y_min = min(prob_response_array[j, :])
                y_max = max(prob_response_array[j, :])

                # Plot asymptotes (commented out) and force to NaN values farther than asymptotes for MMA_based
                for intv in subprob.approx.interv:
                    if intv.__class__.__name__ == 'MMA':
                        # L_i = plt.axvline(x=intv[0].low[i], color='g', label=f'$L_{i}^{{(k)}}$')
                        # U_i = plt.axvline(x=intv[0].upp[i], color='y', label=f'$U_{i}^{{(k)}}$')

                        # Put = NaN the points of g_j_tilde that x_i > U_i and x_i < L_i
                        for k in range(0, self.x.shape[1]):
                            if (self.x[i, k] <= 1.01 * intv.low[i]) or (self.x[i, k] >= 0.99 * intv.upp[i]):
                                approx_response_array[j, k] = np.NaN
                    elif intv.__class__.__name__ == 'Reciprocal':
                        for k in range(0, self.x.shape[1]):
                            if self.x[i, k] <= 0:
                                approx_response_array[j, k] = np.NaN

                # First plot
                if self.iter_pair == 0:

                    # Plot Taylor expansion point
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$x^{(k)}$' + '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='o', markersize=9)

                    # Plot exact response
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'k',
                                          label='$g~[x]$')

                    # Plot approximate response
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :],
                                            'b--', label=r'$\tilde{g}_{lin}~[x]$')

                # Second plot
                elif self.iter_pair == 1:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :],
                                            'r--', label=r'$\tilde{g}_{T2}~[x]$')

                # Third plot
                elif self.iter_pair == 2:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :],
                                            'y--', label=r'$\tilde{g}_{rec}~[x]$')

                # Fourth plot
                elif self.iter_pair == 3:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :],
                                            'c--', label=r'$\tilde{g}_{mma}~[x]$')

                # Plot details (labels, title, x_limit, y_limit, fontsize, legend, etc)
                x_min = self.x[i, 0].copy()
                x_max = self.x[i, -1].copy()

                # Print approximation name for each pair of {x_i, g_j} and set limits for x and y axes
                ax.set(xlabel='$x$', ylabel='$g$',
                       # xlim=(x_min - 0.01 * (x_max - x_min), x_max + 0.01 * (x_max - x_min)),
                       ylim=(y_min - 0.01 * (y_max - y_min), y_max + 0.01 * (y_max - y_min)))

                # FontSize for title, xlabel and ylabel set to 20
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                    item.set_fontsize(20)
                plt.grid(True)
                plt.legend(loc='upper left')
                plt.show(block=False)

        self.iter_pair += 1


def plot_approx():
    logger.info("Solving test_poly using y=MMA and solver=Ipopt Svanberg")

    # Instantiate problem
    prob = Polynomial1D()

    # Instantiate subproblems
    subprob1 = sao.Subproblem(approximation=sao.Taylor1(sao.Linear()))
    subprob2 = sao.Subproblem(approximation=sao.Taylor2(sao.Linear()))
    subprob3 = sao.Subproblem(approximation=sao.Taylor1(sao.Reciprocal()))
    subprob4 = sao.Subproblem(approximation=sao.Taylor1(sao.MMA(prob.xmin, prob.xmax)))

    # Instantiate plotter
    plotter = PlotApprox(prob, variables=np.array([0]), responses=np.array([0]))

    # Initialize iteration counter and design
    itte = 0
    x_k = np.array([0.5])

    # Evaluate responses and sensitivities at current point, i.e. g(X^(k)), dg(X^(k)), ddg(X^(k))
    f = prob.g(x_k)
    df = prob.dg(x_k)
    ddf = prob.ddg(x_k)

    # Build approximate sub-problem at X^(k)
    subprob1.build(x_k, f, df)
    subprob2.build(x_k, f, df, ddf)
    subprob3.build(x_k, f, df)
    subprob4.build(x_k, f, df)

    # Plot current approximation
    plotter.plot_approximation(x_k, f, prob, subprob1)
    plotter.plot_approximation(x_k, f, prob, subprob2)
    plotter.plot_approximation(x_k, f, prob, subprob3)
    plotter.plot_approximation(x_k, f, prob, subprob4)

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    plot_approx()
