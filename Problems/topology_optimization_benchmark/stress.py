from .mbbbeam import MBBBeam
import numpy as np


class Stress(MBBBeam):
    def __init__(self, nelx, nely, volfrac, penal, rmin, max_stress=10):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.max_stress = max_stress
        self.name = 'Stress'
        self.P = 6
        self.iter = 1

        # Solution and RHS vectors
        self.f = np.zeros((self.ndof, 1))
        self.u = np.zeros((self.ndof, 1))
        self.stress = np.zeros(self.n)


        # Set load
        self.f[self.dout, 0] = -1

        self.B = 0.5 * np.array(
            [[-1, 0, 1, 0, 1, 0, -1, 0],
             [0, -1, 0, -1, 0, 1, 0, 1],
             [-1, -1, -1, 1, 1, 1, 1, -1]])

        nu = 0.3
        self.D = 1 / ((1 + nu) * (1 - 2 * nu)) * np.array(
            [[1 - nu, nu, 0],
             [nu, 1 - nu, 0],
             [0, 0, (1 - 2 * nu) / 2]])

        self.V = np.array([[1, -0.5, 0],
                           [-0.5, 1, 0],
                           [0, 0, 3]])

        self.iter = False
        self.factor = 0

    def g(self, x):

        g_j = np.empty(self.m + 1)

        # Filter design variables
        xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        self.K = self.assemble_K(xPhys)
        self.u[self.free, :] = self.linear_solve(self.K, self.f[self.free, :])

        self.elemental_strain = self.B.dot(self.u.flatten()[self.edofMat].transpose())
        self.elemental_strain[2, :] *= 2  # voigt notation

        self.elemental_stress = self.D.dot(self.elemental_strain).transpose()

        self.stress_vm0 = (self.elemental_stress.dot(self.V) * self.elemental_stress).sum(1)
        self.stress_vm = np.sqrt(self.stress_vm0)

        self.stress[:] = xPhys * self.stress_vm.flatten()
        self.gi = (self.stress_vm / self.max_stress) - 1
        self.gi_scaled = xPhys * self.gi
        self.giplus = self.gi_scaled + 1
        self.giP = self.giplus**self.P
        self.gisum = (1/self.n) * np.sum(self.giP)
        giPP = self.gisum**(1/self.P)
        g_j[1] = giPP - 1
        if self.iter == False:
            self.factor = sum(xPhys[:]) / self.n
            self.iter = True

        g_j[0] = 100 * (sum(xPhys[:]) / self.n)/self.factor
        return g_j

    def dg(self, x_k):
        dg_j = np.empty((self.m + 1, self.n))

        # Filter design variable sensitivities
        # TODO unfortunately we filter twice (both in g and dg), can we circumvent this?
        xPhys = np.asarray(self.H * x_k[np.newaxis].T / self.Hs)[:, 0]

        dgdgi_scaled = (1/self.n) * self.gisum**(1/self.P - 1) * self.giplus**(self.P - 1)
        dgidstress = dgdgi_scaled[:, np.newaxis] * xPhys[:, np.newaxis] * (self.stress_vm0**(-0.5)/self.max_stress)[:, np.newaxis] * self.elemental_stress.dot(self.V)
        dgdsstrainmat = np.einsum('jk,kl->jl', dgidstress, self.D)
        dgdsstrainmat[:, 2] *= 2
        dgdue = np.einsum('ij,jl->il', dgdsstrainmat, self.B)
        y = np.zeros(self.ndof)
        for i in range(0, self.n):
            y[self.edofMat[i, :]] += dgdue[i, :]

        lag = np.zeros((self.ndof,1))
        lag[self.free,:] = self.linear_solve(self.K, y[self.free])
        ce = (np.dot(self.u[self.edofMat].reshape(self.nelx * self.nely, 8), self.KE) *
                      lag[self.edofMat].reshape(self.nelx * self.nely, 8)).sum(1)

        bro = (-self.penal * xPhys ** (self.penal - 1) * (1 - self.Eps)) * ce

        dg_j[1,:] = bro
        dg_j[1,:] += (dgdgi_scaled * self.gi)

        dg_j[0, :] = 100 * (np.ones(self.n) / (self.n))/self.factor

        # Sensitivity filtering
        dg_j[0, :] = np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j


if __name__ == "__main__":
    prob = Stress(8, 3, 0.6, 3, 2)
    x = np.random.rand(prob.n)*1.0
    g0 = prob.g(x)
    dg_an = prob.dg(x)

    dx = 1e-4
    dg_fd = np.zeros_like(dg_an)
    for i in range(prob.n):
        x0 = x[i]
        x[i] += dx
        gp = prob.g(x)
        x[i] = x0
        dg_fd[:, i] = (gp - g0) / dx
        print(f"an: {dg_an[:, i]}, fd: {dg_fd[:, i]}, diff = {dg_an[:, i]/dg_fd[:, i] - 1.0}")

