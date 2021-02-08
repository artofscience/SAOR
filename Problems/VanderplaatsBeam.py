## IMPORTS
import numpy as np
import matplotlib.pyplot as plt


## CLASS: This is the Vanderplaats cantilever beam by Dirk
class Vanderplaats:

    def __init__(self, N):
        self.name = 'Vanderplaats'

        # Number of segments
        self.N = N

        # Number of variables: b,h of each segment
        self.n = 2 * N

        # Number of responses: stress, aspect ratio displacement constraints
        self.m = 2 * N + 1

        # Initial point
        self.x_init = np.zeros(self.n, dtype=np.float_)
        self.x_init[:N] = 5e0                                      # initial -b-
        self.x_init[N:] = 40e0                                     # initial -h-

        # Variable bounds
        self.xmin = np.ones(self.n, dtype=np.float_) * 1e-1        # self.xmin
        self.xmax = np.ones(self.n, dtype=np.float_) * 1e2         # self.xmax

        # Stress and displacement limits
        self.sig_max = 14e3
        self.y_max = 2.5

        # Constraint right-hand-sides
        c_r = np.zeros(self.m, dtype=np.float_)
        c_r[:N] = self.sig_max
        c_r[N:2 * N] = 0e0
        c_r[2 * N] = self.y_max

        # # Constraint sense
        # c_s = np.zeros(self.m, dtype=np.int_)
        # cnv = 1e-2
        # mov = 0.1
        # mxi = 999

        # Parameters
        self.P = 5e4                    # Load
        self.E = 2e7                    # Young's Modulus
        self.L = 5e2                    # Total length
        self.S = self.L / self.N        # Segment length

    def response(self, x_k):
        g_j = np.zeros(self.m + 1, dtype=np.float_)
        for i in range(self.N):
            b = x_k[i]                   # get width
            h = x_k[self.N + i]          # get height

            # Weight objective
            g_j[0] = g_j[0] + self.S * b * h

            # Force moment
            M = self.P * (self.L - float(i + 1) * self.S + self.S)

            # Second moment of area
            I = b * h ** 3 / 12

            # Stress constraint
            sts = (M * h) / (2 * I)
            g_j[1 + i] = sts / self.sig_max - 1.

            # Geometric constraint
            g_j[1 + self.N + i] = h - 20 * b

        # Displacement constraints
        y = 0e0
        ya = 0e0
        for i in range(self.N):
            b = x_k[i]                  # get width
            h = x_k[self.N + i]         # get height

            # Second moment of area
            I = b * h ** 3e0 / 12e0

            # Left displacement
            y = (self.P * self.S ** 2) / (2 * self.E * I ) * (self.L - (i + 1) * self.S + 2 * self.S / 3) + ya * self.S + y

            # Right displacement
            ya = (self.P * self.S) / (self.E * I) * (self.L - (i + 1) * self.S + self.S / 2) + ya

            # Displacement constraint value
            g_j[1 + 2 * self.N] = y / self.y_max - 1.

        return g_j

    def sensitivity(self, x_k):
        dg_j = np.zeros((self.m + 1, self.n), dtype=np.float_)
        for i in range(self.N):
            b = x_k[i]                      # get width
            h = x_k[self.N + i]             # get height

            # Derivatives of objective
            dg_j[0, i] = self.S * h
            dg_j[0, self.N + i] = self.S * b

            # Force moment
            M = self.P * (self.L - (i + 1) * self.S + self.S)

            # Second moment of area and its derivatives
            I = b * h ** 3 / 12
            dIdb = h ** 3 / 12
            dIdh = 3 * b * h ** 2 / 12

            # Stress constraint sensitivities
            sts = (M * h) / (2 * I)
            dg_j[1 + i, i] = -sts / I * dIdb / self.sig_max
            dg_j[1 + i, self.N + i] = -12 * M / b / h ** 3 / self.sig_max

            # Geometric constraint sensitivities
            dg_j[1 + self.N + i][i] = -20.
            dg_j[1 + self.N + i][self.N + i] = 1.

        # Displacement constraint sensitivities
        y = 0e0
        ya = 0e0
        for i in range(self.N):
            b = x_k[i]                          # get width
            h = x_k[self.N + i]                 # get height

            # Second moment of area and its derivatives
            I = b * h ** 3e0 / 12e0
            dIdb = h ** 3e0 / 12e0
            dIdh = 3e0 * b * h ** 2e0 / 12e0

            # Left displacement
            y = (self.P * self.S ** 2) / (2 * self.E * I) * (self.L - (i + 1) * self.S + 2 * self.S / 3) + ya * self.S + y

            # Right displacement
            ya = (self.P * self.S) / (self.E * I) * (self.L - (i + 1) * self.S + self.S / 2) + ya

            # The derivatives are a function of this segment
            dA_db = -self.P * self.S ** 2 / self.E / I / I / 2 * (self.L - (i + 1) * self.S + 2 * self.S / 3) * dIdb / self.y_max
            dA_dh = -self.P * self.S ** 2 / self.E / I / I / 2 * (self.L - (i + 1) * self.S + 2 * self.S / 3) * dIdh / self.y_max

            # and the previous
            dAa_db = -self.P * self.S / self.E / I / I * (self.L - (i + 1) * self.S + self.S / 2) * dIdb / self.y_max
            dAa_dh = -self.P * self.S / self.E / I / I * (self.L - (i + 1) * self.S + self.S / 2) * dIdh / self.y_max

            # which gives
            dg_j[1 + 2 * self.N][i] = (self.N - i - 1) * dAa_db * self.S + dA_db
            dg_j[1 + 2 * self.N][self.N + i] = (self.N - i - 1) * dAa_dh * self.S + dA_dh

        return dg_j

    # Plot function
    def kout(self, k, t, vis, x_k):
        s = int(200 / self.N)
        x = np.arange(0, s * self.N, s)
        y = np.zeros(self.N)

        if t != 1:
            if k == 0:
                plt.ion()
                fig, (ax1, ax2) = plt.subplots(2, 1)
                ax1.set_xlim([0, 200e0])
                ax1.set_ylim([-50e0, 50e0])
                ax2.set_xlim([0, 200e0])
                ax2.set_ylim([-5e0, 5e0])
                ims1 = []
                ims2 = []
                for i in range(self.N):
                    im1 = plt.Rectangle((x[i], -x_k[self.N + i] / 2e0), s, x_k[self.N + i], fill=None)
                    im2 = plt.Rectangle((x[i], -x_k[i] / 2e0), s, x_k[i], fill=None)
                    ax1.add_patch(im1)
                    ims1.append(im1)
                    ax2.add_patch(im2)
                    ims2.append(im2)
                fig.show()
                vis = [fig, ax1, ax2, ims1, ims2]
                return vis
            else:
                ims1 = vis[3]
                ims2 = vis[4]
                fig = vis[0]
                for i in range(self.N):
                    ims1[i].set_height(x_k[self.N + i])
                    ims1[i].set_y(-x_k[self.N + i] / 2e0)
                    ims2[i].set_height(x_k[i])
                    ims2[i].set_y(-x_k[i] / 2e0)
                fig.canvas.draw()
                return vis
        else:
            ims1 = vis[3]
            ims2 = vis[4]
            fig = vis[0]
            for i in range(self.N):
                ims1[i].set_height(x_k[self.N + i])
                ims1[i].set_y(-x_k[self.N + i] / 2e0)
                ims2[i].set_height(x_k[i])
                ims2[i].set_y(-x_k[i] / 2e0)
            fig.canvas.draw()
            fig.savefig('vds.png')
            return vis
