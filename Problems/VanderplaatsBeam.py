## IMPORTS
from sao.problems.problem import Problem
import numpy as np
import matplotlib.pyplot as plt


## CLASS: This is the Vanderplaats cantilever beam by Dirk (scaled objective as: g_0' = self.scale * g_0)
class Vanderplaats(Problem):

    def __init__(self, N):
        super().__init__()

        # Number of segments
        self.N = N

        # Scaling factor
        self.scale = 1e-2

        # Number of variables: b,h of each segment
        self.n = 2 * self.N

        # Number of responses: stress, aspect ratio and tip displacement constraints
        self.m = 2 * self.N + 1

        # Initial point
        self.x0 = np.zeros(self.n, dtype=float)
        self.x0[:self.N] = 5.       # initial -b-
        self.x0[self.N:] = 40.      # initial -h-

        # Variable bounds
        self.xmin = np.ones(self.n, dtype=float) * 1e-1         # self.xmin
        self.xmax = np.ones(self.n, dtype=float) * 1e2          # self.xmax

        # Parameters
        self.sig_max = 14e3  # Stress limit
        self.y_max = 2.5  # Displacement limit
        self.P = 5e4  # Load
        self.E = 2e7  # Young's Modulus
        self.L = 5e2  # Total length
        self.S = self.L / self.N  # Segment length

    def g(self, x_k):
        y = 0.
        ya = 0.
        g = np.zeros((self.m + 1), dtype=float)

        for i in range(self.N):
            b = x_k[i]                  # get width
            h = x_k[self.N + i]         # get height

            # Weight objective
            g[0] = g[0] + self.S * b * h * self.scale

            # Force moment
            M = self.P * (self.L - (i + 1) * self.S + self.S)

            # Second moment of area
            I = b * h ** 3 / 12

            # Stress constraints
            sts = (M * h) / (2 * I)
            g[1 + i] = sts / self.sig_max - 1.

            # Geometric constraints
            g[1 + self.N + i] = h - 20 * b

            # Left displacement
            y = (self.P * self.S ** 2) / (2 * self.E * I) * (
                        self.L - (i + 1) * self.S + 2 * self.S / 3) + ya * self.S + y

            # Right displacement
            ya = (self.P * self.S) / (self.E * I) * (self.L - (i + 1) * self.S + self.S / 2) + ya

        # Displacement constraint value
        g[1 + 2 * self.N] = y / self.y_max - 1

        return g

    def dg(self, x_k):
        y = 0.
        ya = 0.
        dg = np.zeros((self.m + 1, self.n), dtype=float)
        for i in range(self.N):
            b = x_k[i]              # get width
            h = x_k[self.N + i]     # get height

            # Derivatives of objective
            dg[0, i] = self.S * h * self.scale
            dg[0, self.N + i] = self.S * b * self.scale

            # Force moment
            M = self.P * (self.L - (i + 1) * self.S + self.S)

            # Second moment of area and its derivatives
            I = b * h ** 3 / 12
            dIdb = h ** 3 / 12
            dIdh = 3 * b * h ** 2 / 12

            # Stress constraint sensitivities
            sts = (M * h) / (2 * I)
            dg[1 + i, i] = - (6 * M) / (self.sig_max * h ** 2 * b ** 2)
            dg[1 + i, self.N + i] = - (12 * M) / (self.sig_max * b * h ** 3)

            # Geometric constraint sensitivities
            dg[1 + self.N + i, i] = -20.
            dg[1 + self.N + i, self.N + i] = 1.

            # Left displacement
            y = (self.P * self.S ** 2) / (2 * self.E * I) * (
                        self.L - (i + 1) * self.S + 2 * self.S / 3) + ya * self.S + y

            # Right displacement
            ya = (self.P * self.S) / (self.E * I) * (self.L - (i + 1) * self.S + self.S / 2) + ya

            # The derivatives are a function of this segment
            dA_db = -self.P * self.S ** 2 / self.E / I / I / 2 * (self.L - (i + 1) * self.S + 2 * self.S / 3) * dIdb
            dA_dh = -self.P * self.S ** 2 / self.E / I / I / 2 * (self.L - (i + 1) * self.S + 2 * self.S / 3) * dIdh

            # and the previous
            dAa_db = -self.P * self.S / self.E / I / I * (self.L - (i + 1) * self.S + self.S / 2) * dIdb
            dAa_dh = -self.P * self.S / self.E / I / I * (self.L - (i + 1) * self.S + self.S / 2) * dIdh

            # Displacement constraint sensitivities
            dg[1 + 2 * self.N, i] = ((self.N - i - 1) * dAa_db * self.S + dA_db) / self.y_max
            dg[1 + 2 * self.N, self.N + i] = ((self.N - i - 1) * dAa_dh * self.S + dA_dh) / self.y_max

        return dg

    def visualize(self, x_k, iteration, vis, **kwargs):
        """Function to visualize current design"""
        s = int(self.L / self.N)
        x = np.arange(0, s * self.N, s)
        t = kwargs.get('t', 0)

        if t != 1:
            if iteration == 0:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(2, 1)
                fig.suptitle('iter = {}'.format(iteration), fontsize=20)
                ax1.set_ylabel('h [cm]', fontsize=16)
                ax2.set_ylabel('b [cm]', fontsize=16)
                ax2.set_xlabel('L [cm]', fontsize=16)
                ax1.set_xlim([0, self.L])
                ax1.set_ylim([-50, 50])
                ax2.set_xlim([0, self.L])
                ax2.set_ylim([-5, 5])
                ims1 = []
                ims2 = []
                for i in range(self.N):
                    im1 = plt.Rectangle((x[i], -x_k[self.N + i] / 2), s, x_k[self.N + i], fill=None)
                    im2 = plt.Rectangle((x[i], -x_k[i] / 2), s, x_k[i], fill=None)
                    ax1.add_patch(im1)
                    ims1.append(im1)
                    ax2.add_patch(im2)
                    ims2.append(im2)
                fig.show()
                vis = [fig, ax1, ax2, ims1, ims2]
                return vis
            else:
                vis[0].suptitle('iter = {}'.format(iteration), fontsize=20)
                ims1 = vis[3]
                ims2 = vis[4]
                fig = vis[0]
                for i in range(self.N):
                    ims1[i].set_height(x_k[self.N + i])
                    ims1[i].set_y(-x_k[self.N + i] / 2)
                    ims2[i].set_height(x_k[i])
                    ims2[i].set_y(-x_k[i] / 2)
                fig.canvas.draw()
                return vis
        else:
            ims1 = vis[3]
            ims2 = vis[4]
            fig = vis[0]
            for i in range(self.N):
                ims1[i].set_height(x_k[self.N + i])
                ims1[i].set_y(-x_k[self.N + i] / 2)
                ims2[i].set_height(x_k[i])
                ims2[i].set_y(-x_k[i] / 2)
            fig.canvas.draw()
            fig.savefig('vds.png')
            return vis
