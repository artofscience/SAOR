
import sys
from Problems.topology_optimization.mbbbeam import MBBBeam
import numpy as np
from scipy.sparse.linalg import eigsh, splu, LinearOperator, spsolve
import matplotlib.pyplot as plt


class DynamicCompliance(MBBBeam):
    def __init__(self, nelx, nely, volfrac=0.6, penal=3, rmin=2, rho=1e-10, compute_eigenvalues=False, objective_scale=100.0, omega=1000.0, plot_frf=False):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.unitL = 10.0/nelx

        # Solution and RHS vectors
        self.u = np.zeros(self.ndof, dtype=np.complex128)
        self.f = np.zeros(self.ndof)
        self.f[self.dout] = -1  # Set load

        # Material density
        self.rho = rho

        # Rayleigh damping coefficients
        self.alpha = 1e-5
        self.beta = 1e-5

        # Evaluation frequency
        self.omega = omega
        self.compute_eigenvalues = compute_eigenvalues

        # Objective scaling_strategies
        self.objective_scale = objective_scale
        self.g0fac = None

        self.xPhys = None
        self.f_compl = 0

        if plot_frf:
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax.set_title("Frequency response function")
            self.ax.set_xlabel("Frequency (rad/s)")
            self.ax.set_ylabel("Amplitude (m/N)")
            self.ax.set_yscale('log')
            plt.show(block=False)
            self.omg_line, = self.ax.plot([], [], 'k--')
            self.omg_dot, = self.ax.plot([], [], 'k.')
            self.frf_line, = self.ax.plot([], [])

        else:
            self.fig, self.ax = None, None

    def g(self, x):
        # Filter design variables
        self.xPhys = np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]

        # Assemble mass and stiffness matrix
        K = self.assemble_K(self.xPhys, interpolation='simplin')
        M = self.assemble_M(self.xPhys, rho=self.rho, lx=self.unitL, ly=self.unitL)

        # Dynamic stiffness matrix
        Z = K + 1j*self.omega*(self.alpha*M + self.beta*K) - self.omega**2 * M

        # Solve the dynamic equation
        self.u[self.free] = spsolve(Z, self.f[self.free])
        self.f_compl = self.u@self.f

        if self.compute_eigenvalues or self.fig is not None:
            Kinv = splu(K)
            KinvOp = LinearOperator(K.shape, Kinv.solve)

            # Solve the eigenvalue problem
            n_modes = 20
            eigvals, eigvecs = eigsh(K, M=M, k=n_modes,  OPinv=KinvOp, sigma=0.0)
            if self.compute_eigenvalues:
                print(f"Eigenfrequencies: {np.sqrt(eigvals)}")
            if self.fig is not None:
                wmin, wmax = 0, 2*self.omega
                wrange = np.linspace(wmin, wmax, 1000)
                # --- Approximate response ---
                u_aug = KinvOp(self.f[self.free])
                for i in range(n_modes):
                    u_aug -= u_aug @ M @ eigvecs[:, i] * eigvecs[:, i]
                u_aug /= u_aug @ M @ u_aug

                V = np.hstack((eigvecs, u_aug[..., np.newaxis]))
                lam_red = np.hstack((eigvals, u_aug@K@u_aug))

                Fr = self.f[self.free]@V
                frf = np.absolute(Fr@(Fr[..., np.newaxis] /
                                      ((lam_red[..., np.newaxis] - (wrange**2)[np.newaxis]) +
                                       1j*wrange[np.newaxis]*(self.alpha + self.beta*lam_red[..., np.newaxis]))))

                '''
                 # --- Actual response ---
                frf = np.zeros_like(wrange)
                for i, w in enumerate(wrange):
                    sys.stdout.write(f"\rSolving frequency {i+1}/{len(wrange)}...")
                    sys.stdout.flush()

                    # Dynamic stiffness matrix
                    Zi = K + 1j*w*(self.alpha*M + self.beta*K) - w**2 * M

                    # Solve the dynamic equation
                    u = spsolve(Zi, self.f[self.free])
                    frf[i] = np.absolute(u@self.f[self.free])
                sys.stdout.write(f"All {len(wrange)} frequencies solved\n")
                sys.stdout.flush()
                '''
                # ---------------------------------

                ymin, ymax = min(frf)*0.9, max(frf)*1.1
                self.frf_line.set_xdata(wrange)
                self.frf_line.set_ydata(frf)
                self.omg_line.set_xdata([self.omega, self.omega])
                self.omg_line.set_ydata([ymin, ymax])
                self.omg_dot.set_xdata([self.omega])
                self.omg_dot.set_ydata([np.absolute(self.f_compl)])
                self.ax.set_ylim([ymin, ymax])
                self.ax.set_xlim([0, 2*self.omega])


                self.fig.canvas.draw()
                self.fig.canvas.flush_events()





        # Calculate responses
        g_j = np.empty(2)
        g_j[0] = np.absolute(self.f_compl)
        if self.g0fac is None:
            self.g0fac = self.objective_scale/g_j[0]
        g_j[0] *= self.g0fac
        g_j[1] = (sum(self.xPhys[:]) / (self.volfrac * self.n) - 1)*10
        return g_j

    def dg(self, x):
        dg_j = np.zeros((2, self.n))

        ue = self.u[self.edofMat]

        dg_dabs = self.f_compl / np.absolute(self.f_compl)

        dg_dK = dg_dabs*(-1+1j*self.omega*self.beta)
        dg_dM = dg_dabs*(self.omega**2 + 1j*self.omega*self.alpha)

        dg_dsK = np.real(dg_dK*np.conj(np.einsum("ij,jk,ik->i", ue, self.KE, ue)))
        m_el = self.rho*self.unitL*self.unitL/4
        dg_dsM = np.real(dg_dM*m_el*np.conj(np.einsum("ij,ij->i", ue, ue)))

        dg_j[0, :] += dg_dsK*(1-self.Eps)*(0.1 + 0.9*self.penal*self.xPhys**(self.penal-1))
        dg_j[0, :] += dg_dsM*(1-self.Eps)

        # Volume
        dg_j[1, :] = np.ones(self.nely * self.nelx) / (self.volfrac * self.n)

        # Sensitivity filtering
        dg_j[0, :] = self.g0fac*np.asarray(self.H * (dg_j[0, :][np.newaxis].T / self.Hs))[:, 0]
        dg_j[1, :] = 10*np.asarray(self.H * (dg_j[1, :][np.newaxis].T / self.Hs))[:, 0]

        return dg_j


if __name__ == "__main__":
    n = 10
    prob = DynamicCompliance(5*n, n)
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


