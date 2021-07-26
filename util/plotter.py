import numpy as np
import matplotlib.pyplot as plt


class Plot2:
    """
    This class is used to generate the plots for `non-MixedMoveLimit` subproblems.
    Includes a method to plot pairs of {g_j - x_i} for low-dimensional problems, as well as a method to generate
    contour plots for 2D problems, i.e. X = [x1, x2].
    """

    def __init__(self, prob, **kwargs):
        """
        :param prob: From the prob object you need to extract the bounds for the variables.
        :param kwargs: From the kwargs you get which pairs of {g_j - x_i} you want to plot. If none are given, all
                       possible combinations are plotted, that is n*(m+1) plots.
        """
        self.iter_pair = 0
        self.iter_contour = 0
        self.x = np.linspace(prob.xmin, prob.xmax, 100).T
        self.resp = kwargs.get('responses', np.arange(0, prob.m + 1))
        self.vars = kwargs.get('variables', np.arange(0, prob.n))
        self.fig = []
        self.fig_idx = {}

    def plot_pair(self, x_k, f, prob, subprob, itte):
        """
        This function plots (some of) the {g_j - x_i}, for j in `responses` & i in `variables`.
        On the same graph, exact plots of {g_j - x_i} and approximate plots of {g_j_tilde - x_i} are shown in order to
        assess the quality of the generated approximation.
        Since all approximations must be separable (for the solver to be efficient), the exact plots show
        the best function approximation one can achieve with such (separable) analytical Taylor-like expansions. Also,
        by studying such plots, one can get a better understanding as to what intervening variables would be beneficial.
        :param x_k: The current design.
        :param f: The current response values.
        :param prob: This object is used to evaluate the exact responses, i.e. prob.g.
        :param subprob: This object is used to get several useful data for the plots, e.g. n, m, g, inter, etc.
        :param itte: Current iteration number to show on the title of the plots.
        :return:
        """

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
                if self.iter_pair > 0:
                    y_min = min(min(prob_response_array[j, :]), ax.get_ylim()[0])
                    y_max = max(max(prob_response_array[j, :]), ax.get_ylim()[1])
                else:
                    y_min = min(prob_response_array[j, :])
                    y_max = max(prob_response_array[j, :])

                # Plot new exact response
                if self.iter_pair % 2 == 1:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'b',
                                          label='$g_{}$'.format({j}) + '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$')
                else:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'r',
                                          label='$g_{}$'.format({j}) + '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$')

                # Plot asymptotes (commented out) and force to NaN values farther than asymptotes for MMA_based
                for intv in subprob.approx.interv:
                    if 'MMA' in intv.__class__.__name__:
                        # L_i = plt.axvline(x=intv[0].low[i], color='g', label=f'$L_{i}^{{(k)}}$')
                        # U_i = plt.axvline(x=intv[0].upp[i], color='y', label=f'$U_{i}^{{(k)}}$')

                        # Put = NaN the points of g_j_tilde that x_i > U_i and x_i < L_i
                        for k in range(0, self.x.shape[1]):
                            if (self.x[i, k] <= 1.001 * intv.low[i]) or (self.x[i, k] >= 0.999 * intv.upp[i]):
                                approx_response_array[j, k] = np.NaN

                # Alternate between red and blue plots to tell them apart easily
                if self.iter_pair % 2 == 1:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'b--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({itte}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i}) +
                                               '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='o', markersize=9)
                    alpha = plt.axvline(x=subprob.alpha[i], color='b', linestyle=(0, (3, 8)),
                                        label=fr'$\alpha_{i}^{{({itte})}}$')
                    beta = plt.axvline(x=subprob.beta[i], color='b', linestyle=(0, (3, 8, 1, 8)),
                                       label=fr'$\beta_{i}^{{({itte})}}$')
                else:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'r--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({itte}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i}) +
                                               '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='s', markersize=9)
                    alpha = plt.axvline(x=subprob.alpha[i], color='r', linestyle=(0, (3, 8)),
                                        label=fr'$\alpha_{i}^{{({itte})}}$')
                    beta = plt.axvline(x=subprob.beta[i], color='r', linestyle=(0, (3, 8, 1, 8)),
                                       label=fr'$\beta_{i}^{{({itte})}}$')

                # Delete the plot for (k-2), i.e. L_ji, U_ji, g_i(X), g_i_tilde(X) & respective legends
                if self.iter_pair > 1:
                    for m in range(0, 5):  # Change 5 to 7 if you wanna plot asymptotes cuz you add 2 more lines
                        ax.lines[0].remove()

                # Plot details (labels, title, x_limit, y_limit, fontsize, legend, etc)
                x_min = self.x[i, 0].copy()
                x_max = self.x[i, -1].copy()

                # Print approximation name for each pair of {x_i, g_j} and set limits for x and y axes
                ax.set(xlabel=f'$x_{i}$', ylabel=f'$g_{j}$',
                       # xlim=(x_min - 0.01 * (x_max - x_min), x_max + 0.01 * (x_max - x_min)),
                       ylim=(y_min - 0.01 * (y_max - y_min), y_max + 0.01 * (y_max - y_min)),
                       title='%s: {} - {} \n  $iter = {}$'.format(subprob.approx.interv[0].__class__.__name__,
                                                                  subprob.approx.__class__.__name__,
                                                                  itte)
                             % prob.__class__.__name__)

                # FontSize for title, xlabel and ylabel set to 20
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                    item.set_fontsize(20)
                plt.grid(True)
                plt.legend(loc='upper right')
                plt.show(block=False)

        self.iter_pair += 1

    def contour_plot(self, x_k, f, prob, subprob, itte):
        """
        This method is used to generate contour plots of {g_j - x_i} for 2D problems, where X = [x1, x2].
        Both the exact P_{NLP} and the approximate P_{NLP}_tilde are plotted at each design iteration.
        :param x_k: The current design.
        :param f: The current response values.
        :param prob: This object is used to evaluate the exact responses, i.e. prob.g.
        :param subprob: This object is used to get several useful data for the plots, e.g. n, m, g, inter, etc.
        :param itte: Current iteration number to show on the title of the plots.
        :return:
        """
        X, Y = np.meshgrid(self.x[0, :], self.x[1, :])

        # For the first iteration
        if self.iter_contour == 0:

            # Exact problem P_nlp: z_exact[m+1, x2, x1] has values for responses g_j(x_curr), for all x1, x2
            z_exact = np.empty((prob.m + 1, self.x.shape[1], self.x.shape[1]))
            x_curr = np.empty((prob.n, 1))
            for k2 in range(0, self.x.shape[1]):  # sweeping x2
                for k1 in range(0, self.x.shape[1]):  # sweeping x1
                    x_curr[0, 0] = self.x[0, k1].copy()
                    x_curr[1, 0] = self.x[1, k2].copy()
                    z_exact[:, k2, k1] = prob.g(x_curr)

            # Plot g_0 and x_k
            self.fig.append(plt.subplots(1, 1)[0])
            self.fig_idx['fig_exact'] = plt.gcf().number
            fig_exact = plt.figure(self.fig_idx['fig_exact'])
            ax_exact = plt.gca()
            obj_exact = plt.contourf(X, Y, z_exact[0, :, :], 50, cmap='jet')
            point_exact = plt.scatter(x_k[0], x_k[1],
                                      label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                            ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                      marker='o', edgecolors='yellow', color='k', s=100)

            # Plot iso-lines of constraints to show feasible region: g_j(X) = 0
            if prob.m > 0:
                for i in range(1, prob.m + 1):
                    constr_exact = plt.contour(X, Y, z_exact[i, :, :], np.array([-0.1, 0.]), cmap='gray')
                    ax_exact.clabel(constr_exact, inline=1, fontsize=10)

            # Figure properties
            ax_exact.set_title('$P_{NLP}$ of %s' % prob.__class__.__name__, fontsize=20)
            ax_exact.set_xlabel('$x_0$', fontsize=18)
            ax_exact.set_ylabel('$x_1$', fontsize=18)
            cbar = fig_exact.colorbar(obj_exact, shrink=0.5, aspect=8)
            cbar.set_label('$g_0(\mathbf{X})$', labelpad=-30, y=1.15, rotation=0, fontsize=18)
            plt.legend()
            plt.show(block=False)  # Use keyword 'block' to override blocking behaviour of debugger

        # For iteration > 0
        else:

            # Plot in Exact Figure the optimal point found by the solver at the last iteration
            plt.figure(self.fig_idx['fig_exact'])
            point_exact = plt.scatter(x_k[0], x_k[1],
                                      label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                            ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                      marker='o', edgecolors='yellow', color='k', s=100)
            plt.legend()
            plt.show(block=False)  # Use keyword 'block' to override blocking behaviour of debugger

            # Plot in Approx Figure the optimal point found by the solver at the last iteration
            plt.figure(self.fig_idx['fig_approx'])
            opt_point = plt.scatter(x_k[0], x_k[1],
                                      label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                            ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                      marker='o', edgecolors='yellow', color='k', s=100)
            plt.legend()
            plt.show(block=False)  # Use keyword 'block' to override blocking behaviour of debugger

        # Approx problem P_nlp_tilde: z_approx[m+1, x2, x1] has values for responses g_j_tilde(x_curr), for all x1, x2
        z_approx = np.empty((prob.m + 1, self.x.shape[1], self.x.shape[1]))
        x_curr = np.empty(prob.n)
        for k2 in range(0, self.x.shape[1]):  # sweeping x2
            for k1 in range(0, self.x.shape[1]):  # sweeping x1
                x_curr[0] = self.x[0, k1]
                x_curr[1] = self.x[1, k2]
                z_approx[:, k2, k1] = subprob.g(x_curr)

        # For MMA family: Force response values farther than asymptotes to NaN
        for intv in subprob.approx.interv:
            if 'MMA' in intv.__class__.__name__:
                for i in range(0, prob.m + 1):  # for every response -g_j-
                    for k2 in range(0, self.x.shape[1]):
                        if (self.x[1, k2] < 1.001 * intv.low[1]) or (self.x[1, k2] > 0.999 * intv.upp[1]):
                            z_approx[:, k2, :] = np.NaN
                    for k1 in range(0, self.x.shape[1]):
                        if (self.x[0, k1] < 1.001 * intv.low[0]) or (self.x[0, k1] > 0.999 * intv.upp[0]):
                            z_approx[:, :, k1] = np.NaN

        # New plot for approximate problem P_nlp_tilde
        self.fig.append(plt.subplots(1, 1)[0])
        self.fig_idx['fig_approx'] = plt.gcf().number
        fig_approx = plt.figure(self.fig_idx['fig_approx'])
        ax_approx = plt.gca()
        obj_approx = plt.contourf(X, Y, z_approx[0, :, :], 50, cmap='jet')

        # Plot iso-lines of constraints to show feasible region: g_j_tilde(X) = 0
        if prob.m > 0:
            for i in range(1, prob.m + 1):
                constr_approx = plt.contour(X, Y, z_approx[i, :, :], np.array([-0.1, 0.]), cmap='gray')
                ax_approx.clabel(constr_approx, inline=1, fontsize=10)

        # Plot approximation point -x_k-
        point_approx = plt.scatter(x_k[0], x_k[1],
                                   label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                         ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                   marker='o', edgecolors='yellow', color='k', s=100)

        # Plot approximate subproblem bounds -alpha- and -beta-
        alpha0 = plt.axvline(x=subprob.alpha[0], color='w', linestyle=(0, (3, 8)), label=r'$\alpha_i$')
        alpha1 = plt.axhline(y=subprob.alpha[1], color='w', linestyle=(0, (3, 8)))
        beta0 = plt.axvline(x=subprob.beta[0], color='w', linestyle=(0, (3, 8, 1, 8)), label=r'$\beta_i$')
        beta1 = plt.axhline(y=subprob.beta[1], color='w', linestyle=(0, (3, 8, 1, 8)))

        # Figure properties
        ax_approx.set_title('$\widetilde{{P}}_{{NLP}}$: {} - {}, iter = {}'.format(
            subprob.approx.interv[0].__class__.__name__,
            subprob.approx.__class__.__name__, self.iter_contour), fontsize=20)
        ax_approx.set_xlabel('$x_0$', fontsize=18)
        ax_approx.set_ylabel('$x_1$', fontsize=18)
        cbar = fig_approx.colorbar(obj_approx, shrink=0.5, aspect=8)
        cbar.set_label('$g_0(\mathbf{X})$', labelpad=-30, y=1.15, rotation=0, fontsize=18)
        plt.legend()
        plt.show(block=False)  # overwrite blocking behaviour of debugger

        self.iter_contour += 1


class Plot3(Plot2):
    """
    Similar to the `Plot2` class, this class is used to generate plots for `MixedMoveLimit` subproblems.
    It includes a method to plot pairs of {g_j - x_i} for low-dimensional problems,
    as well as a method to generate contour plots for 2D problems, i.e. X = [x1, x2].
    """

    def plot_pair(self, x_k, f, prob, subprob, itte):
        """
        This function plots (some of) the {g_j - x_i}, for j in `responses` & i in `variables`.
        On the same graph, exact plots of {g_j - x_i} and approximate plots of {g_j_tilde - x_i} are shown in order to
        assess the quality of the generated approximation.
        Since all approximations must be separable (for the solver to be efficient), the exact plots show
        the best function approximation one can achieve with such (separable) analytical Taylor-like expansions. Also,
        by studying such plots, one can get a better understanding as to what intervening variables would be beneficial.
        :param x_k: The current design.
        :param f: The current response values.
        :param prob: This object is used to evaluate the exact responses, i.e. prob.g.
        :param subprob: This object is used to get several useful data for the plots, e.g. n, m, g, inter, etc.
        :param itte: Current iteration number to show on the title of the plots.
        :return:
        """

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
                if self.iter_pair > 0:
                    y_min = min(min(prob_response_array[j, :]), ax.get_ylim()[0])
                    y_max = max(max(prob_response_array[j, :]), ax.get_ylim()[1])
                else:
                    y_min = min(prob_response_array[j, :])
                    y_max = max(prob_response_array[j, :])

                # Plot new exact response
                if self.iter_pair % 2 == 1:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'b',
                                          label='$g_{}$'.format({j}) + '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$')
                else:
                    exact_resp = plt.plot(self.x[i, :], prob_response_array[j, :], 'r',
                                          label='$g_{}$'.format({j}) + '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$')

                # Plot asymptotes (commented out) and force to NaN values farther than asymptotes for MMA_based
                for intv in subprob.approx.interv[0].intervening_variables:
                    if 'MMA' in intv.__class__.__name__:
                        # L_i = plt.axvline(x=intv[0].low[i], color='g', label=f'$L_{i}^{{(k)}}$')
                        # U_i = plt.axvline(x=intv[0].upp[i], color='y', label=f'$U_{i}^{{(k)}}$')

                        # Put = NaN the points of g_j_tilde that x_i > U_i and x_i < L_i
                        for k in range(0, self.x.shape[1]):
                            if (self.x[i, k] <= 1.001 * intv.low[i]) or (self.x[i, k] >= 0.999 * intv.upp[i]):
                                approx_response_array[j, k] = np.NaN

                # Alternate between red and blue plots to tell them apart easily
                if self.iter_pair % 2 == 1:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'b--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({itte}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i}) +
                                               '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='o', markersize=9)
                    alpha = plt.axvline(x=subprob.alpha[i], color='b', linestyle=(0, (3, 8)),
                                        label=fr'$\alpha_{i}^{{({itte})}}$')
                    beta = plt.axvline(x=subprob.beta[i], color='b', linestyle=(0, (3, 8, 1, 8)),
                                       label=fr'$\beta_{i}^{{({itte})}}$')
                else:
                    approx_resp, = plt.plot(self.x[i, :], approx_response_array[j, :], 'r--',
                                            label='$\widetilde{g}$' + '$_{}$'.format({j}) + '$^{(}$' +
                                                  '$^{}$'.format({itte}) + '$^{)}$')
                    exp_point = plt.plot(x_k[i], f[j],
                                         label='$X_{}$'.format({i}) +
                                               '$^{(}$' + '$^{}$'.format({itte}) + '$^{)}$' +
                                               '$ = {}$'.format(np.around(x_k[i], decimals=4)),
                                         color='k', marker='s', markersize=9)
                    alpha = plt.axvline(x=subprob.alpha[i], color='r', linestyle=(0, (3, 8)),
                                        label=fr'$\alpha_{i}^{{({itte})}}$')
                    beta = plt.axvline(x=subprob.beta[i], color='r', linestyle=(0, (3, 8, 1, 8)),
                                       label=fr'$\beta_{i}^{{({itte})}}$')

                # Delete the plot for (k-2), i.e. L_ji, U_ji, g_i(X), g_i_tilde(X) & respective legends
                if self.iter_pair > 1:
                    for m in range(0, 5):  # Change 5 to 7 if you wanna plot asymptotes cuz you add 2 more lines
                        ax.lines[0].remove()

                # Plot details (labels, title, x_limit, y_limit, fontsize, legend, etc)
                x_min = self.x[i, 0].copy()
                x_max = self.x[i, -1].copy()

                # Print approximation name for each pair of {x_i, g_j} and set limits for x and y axes
                ax.set(xlabel=f'$x_{i}$', ylabel=f'$g_{j}$',
                       # xlim=(x_min - 0.01 * (x_max - x_min), x_max + 0.01 * (x_max - x_min)),
                       ylim=(y_min - 0.01 * (y_max - y_min), y_max + 0.01 * (y_max - y_min)),
                       title='%s: {} - {} \n  $iter = {}$'.format(subprob.approx.interv[0].__class__.__name__,
                                                                  subprob.approx.__class__.__name__,
                                                                  itte)
                             % prob.__class__.__name__)

                # FontSize for title, xlabel and ylabel set to 20
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                    item.set_fontsize(20)
                plt.grid(True)
                plt.legend(loc='upper right')
                plt.show(block=False)

        self.iter_pair += 1

    def contour_plot(self, x_k, f, prob, subprob, itte):
        """
        This method is used to generate contour plots of {g_j - x_i} for 2D problems, where X = [x1, x2].
        Both the exact P_{NLP} and the approximate P_{NLP}_tilde are plotted at each design iteration.
        It should be used when `MixedMoveLimit` sub-problems are generated.
        :param x_k: The current design.
        :param f: The current response values.
        :param prob: This object is used to evaluate the exact responses, i.e. prob.g.
        :param subprob: This object is used to get several useful data for the plots, e.g. n, m, g, inter, etc.
        :param itte: Current iteration number to show on the title of the plots.
        :return:
        """
        X, Y = np.meshgrid(self.x[0, :], self.x[1, :])

        # For the first iteration
        if self.iter_contour == 0:

            # Exact problem P_nlp: z_exact[m+1, x2, x1] has values for responses g_j(x_curr), for all x1, x2
            z_exact = np.empty((prob.m + 1, self.x.shape[1], self.x.shape[1]))
            x_curr = np.empty((prob.n, 1))
            for k2 in range(0, self.x.shape[1]):  # sweeping x2
                for k1 in range(0, self.x.shape[1]):  # sweeping x1
                    x_curr[0, 0] = self.x[0, k1].copy()
                    x_curr[1, 0] = self.x[1, k2].copy()
                    z_exact[:, k2, k1] = prob.g(x_curr)

            # Plot g_0 and x_k
            self.fig.append(plt.subplots(1, 1)[0])
            self.fig_idx['fig_exact'] = plt.gcf().number
            fig_exact = plt.figure(self.fig_idx['fig_exact'])
            ax_exact = plt.gca()
            obj_exact = plt.contourf(X, Y, z_exact[0, :, :], 50, cmap='jet')
            point_exact = plt.scatter(x_k[0], x_k[1],
                                      label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                            ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                      marker='o', edgecolors='yellow', color='k', s=100)

            # Plot iso-lines of constraints to show feasible region: g_j(X) = 0
            if prob.m > 0:
                for i in range(1, prob.m + 1):
                    constr_exact = plt.contour(X, Y, z_exact[i, :, :], np.array([-0.1, 0.]), cmap='gray')
                    ax_exact.clabel(constr_exact, inline=1, fontsize=10)

            # Figure properties
            ax_exact.set_title('$P_{NLP}$ of %s' % prob.__class__.__name__, fontsize=20)
            ax_exact.set_xlabel('$x_0$', fontsize=18)
            ax_exact.set_ylabel('$x_1$', fontsize=18)
            cbar = fig_exact.colorbar(obj_exact, shrink=0.5, aspect=8)
            cbar.set_label('$g_0(\mathbf{X})$', labelpad=-30, y=1.15, rotation=0, fontsize=18)
            plt.legend()
            plt.show(block=False)  # Use keyword 'block' to override blocking behaviour of debugger

        # For iteration > 0
        else:

            # Plot in Exact Figure the optimal point found by the solver at the last iteration
            plt.figure(self.fig_idx['fig_exact'])
            point_exact = plt.scatter(x_k[0], x_k[1],
                                      label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                            ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                      marker='o', edgecolors='yellow', color='k', s=100)
            plt.legend()
            plt.show(block=False)  # Use keyword 'block' to override blocking behaviour of debugger

            # Plot in Approx Figure the optimal point found by the solver at the last iteration
            plt.figure(self.fig_idx['fig_approx'])
            opt_point = plt.scatter(x_k[0], x_k[1],
                                    label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                          ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                    marker='o', edgecolors='yellow', color='k', s=100)
            plt.legend()
            plt.show(block=False)  # Use keyword 'block' to override blocking behaviour of debugger

        # Approx problem P_nlp_tilde: z_approx[m+1, x2, x1] has values for responses g_j_tilde(x_curr), for all x1, x2
        z_approx = np.empty((prob.m + 1, self.x.shape[1], self.x.shape[1]))
        x_curr = np.empty(prob.n)
        for k2 in range(0, self.x.shape[1]):  # sweeping x2
            for k1 in range(0, self.x.shape[1]):  # sweeping x1
                x_curr[0] = self.x[0, k1]
                x_curr[1] = self.x[1, k2]
                z_approx[:, k2, k1] = subprob.g(x_curr)

        # For MMA family: Force response values farther than asymptotes to NaN
        for intv in subprob.approx.interv[0].intervening_variables:
            if 'MMA' in intv.__class__.__name__:
                # for i in range(0, subprob[p, l].m + 1):
                for k2 in range(0, self.x.shape[1]):
                    if (self.x[1, k2] < 1.001 * intv.low[1]) or (self.x[1, k2] > 0.999 * intv.upp[1]):
                        z_approx[:, k2, :] = np.NaN
                for k1 in range(0, self.x.shape[1]):
                    if (self.x[0, k1] < 1.001 * intv.low[0]) or (self.x[0, k1] > 0.999 * intv.upp[0]):
                        z_approx[:, :, k1] = np.NaN

        # New plot for approximate problem P_nlp_tilde
        self.fig.append(plt.subplots(1, 1)[0])
        self.fig_idx['fig_approx'] = plt.gcf().number
        fig_approx = plt.figure(self.fig_idx['fig_approx'])
        ax_approx = plt.gca()
        obj_approx = plt.contourf(X, Y, z_approx[0, :, :], 50, cmap='jet')

        # Plot iso-lines of constraints to show feasible region: g_j_tilde(X) = 0
        if prob.m > 0:
            for i in range(1, prob.m + 1):
                constr_approx = plt.contour(X, Y, z_approx[i, :, :], np.array([-0.1, 0.]), cmap='gray')
                ax_approx.clabel(constr_approx, inline=1, fontsize=10)

        # Plot approximation point -x_k-
        point_approx = plt.scatter(x_k[0], x_k[1],
                                   label='$\mathbf{X}$' + '$^{}$'.format({self.iter_contour}) +
                                         ' = {d}$^T$'.format(d=np.around(x_k[:], decimals=4)),
                                   marker='o', edgecolors='yellow', color='k', s=100)

        # Plot approximate subproblem bounds -alpha- and -beta-
        alpha0 = plt.axvline(x=subprob.alpha[0], color='w', linestyle=(0, (3, 8)), label=r'$\alpha_i$')
        alpha1 = plt.axhline(y=subprob.alpha[1], color='w', linestyle=(0, (3, 8)))
        beta0 = plt.axvline(x=subprob.beta[0], color='w', linestyle=(0, (3, 8, 1, 8)), label=r'$\beta_i$')
        beta1 = plt.axhline(y=subprob.beta[1], color='w', linestyle=(0, (3, 8, 1, 8)))

        # Figure properties
        ax_approx.set_title('$\widetilde{{P}}_{{NLP}}$: {} - {}, iter = {}'.format(
            subprob.approx.interv[0].__class__.__name__,
            subprob.approx.__class__.__name__, self.iter_contour), fontsize=20)
        ax_approx.set_xlabel('$x_0$', fontsize=18)
        ax_approx.set_ylabel('$x_1$', fontsize=18)
        cbar = fig_approx.colorbar(obj_approx, shrink=0.5, aspect=8)
        cbar.set_label('$g_0(\mathbf{X})$', labelpad=-30, y=1.15, rotation=0, fontsize=18)
        plt.legend()
        plt.show(block=False)  # overwrite blocking behaviour of debugger

        self.iter_contour += 1
