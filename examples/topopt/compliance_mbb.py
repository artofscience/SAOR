import numpy as np
from matplotlib import colors

from Problems.topology_optimization import utils
from Problems.topology_optimization.compliance import ComplianceMBB
import matplotlib.pyplot as plt

from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds, MoveLimit
from sao.intervening_variables import Linear, MMA, MixedIntervening
from sao.approximations import Taylor1
from sao.solvers.primal_dual_interior_point import pdip
from sao.convergence_criteria import IterationCount




def optimize(problem, sub_problem, x, itercount):
    converged = IterationCount(itercount)
    counter = 0
    output = np.zeros((itercount - 1, 5), dtype=float)
    output2 = np.zeros((itercount - 1, 5), dtype=float)
    # setup for plotting design
    plotter = utils.PlotDesign(problem, x)

    f = [0, 0]
    xold = np.zeros_like(x)
    change = np.zeros_like(x)
    expected_lin = 1
    expected_approx = 1

    while not converged:

        plotter.plot(x, counter)

        fold = f[0]
        ## RESPONSES
        f = problem.g(x)
        df = problem.dg(x)
        ##

        # # Scaling
        # if counter == 0:
        #     problem.factor = f[0]
        # f[0] = f[0] / problem.factor
        # df[0, :] = df[0, :] / problem.factor

        print(counter, ":  ", f[0])
        output[counter, 0:2] = f[:]  # objective and constraint values
        output[counter, 2] = np.abs(f[0] - fold)  # objective change
        output2[counter, 0] = (f[0] - fold) / expected_lin  # relative objective change
        output2[counter, 1] = (f[0] - fold) / expected_approx

        xphys = problem.filter.forward(x)
        output2[counter, 2] = np.sum(4 * xphys.flatten() * (1 - xphys.flatten()) / problem.mesh.n)

        ## BUILD SUBPROBLEM
        sub_problem.build(x, f, df)
        ##

        xold[:] = x[:]
        ## SOLVE SUBPROBLEM
        x[:], no_inner_iter = pdip(sub_problem)
        ##
        output2[counter, 3] = no_inner_iter  # number of inner iterations in pdip

        changeold = change
        change = x - xold
        expected_lin = np.dot(df[0, :], change)
        expected_approx = sub_problem.g(x)[0] - sub_problem.g(xold)[0]

        output[counter, 3] = np.mean(np.abs(change))  # mean variable change
        output[counter, 4] = np.max(np.abs(change))  # max variable change
        output2[counter, 4] = np.dot(changeold, change) / \
                              np.dot(np.linalg.norm(changeold), np.linalg.norm(change))


        counter += 1

    output[0, [2, 3, 4]] = np.nan
    output2[0, [0, 1, 4]] = np.nan

    return output, output2, xphys


itercount = 30
x0 = 0.5
nelx = 100
nely = 50

## SETUP SUBPROBLEMS

mma = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1)])
mma.set_name("MMA_asyinit_0.2")

mma_ml = Subproblem(Taylor1(MMA()), limits=[Bounds(0, 1), MoveLimit(0.2)])
mma_ml.set_name("MMA_asyinit_0.2_ML_0.3")

lin_aml = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.2)])
lin_aml.set_name("LIN_AML_0.3")

mix_int = MixedIntervening(nelx * nely, 2, default=Linear())
mix_int.set_intervening(MMA(), resp=0)
mix_mma_lin = Subproblem(Taylor1(mix_int), limits=[Bounds(0, 1)])
mix_mma_lin.set_name("MIX_MMA_asyinit0.2_LIN")

sub_problems = [mma, mma_ml, lin_aml, mix_mma_lin]

figdes, axsdes = plt.subplots(len(sub_problems), sharex=True)

fig, axs = plt.subplots(5, 2, sharex=True)
x = range(0, itercount - 1)

for i, sub_problem in enumerate(sub_problems):
    problem = ComplianceMBB(nelx, nely, vf=x0)
    od1, od2, xphys = optimize(problem, sub_problem, problem.x0, itercount)
    axsdes[i].imshow(-xphys.reshape((problem.mesh.nelx, problem.mesh.nely)).T, cmap='gray',
                     interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    axsdes[i].axis('off')
    for j, dat in enumerate(od1.transpose()):
        axs[j, 0].plot(x, dat)
    for j, dat in enumerate(od2.transpose()):
        axs[j, 1].plot(x, dat)

axs[0, 0].legend([x.name for x in sub_problems])
axs[0, 0].set_ylabel(r'$g_0\left[\mathbf{x}^{(k)}\right]$')

axs[1, 0].set_ylabel(r'$g_1\left[\mathbf{x}^{(k)}\right]$')
axs[1, 0].set_yscale('symlog', linthresh=1e-4)

axs[2, 0].set_ylabel(r'$\left|g_0\left[\mathbf{x}^{(k)}\right] - g_0\left[\mathbf{x}^{(k-1)}\right]\right|$')
axs[2, 0].set_yscale('log')

axs[3, 0].set_ylabel(r'$\frac{1}{n}\sum_{i}^n \left| x_i^{(k)} - x_i^{(k-1)}\right|$')
axs[3, 0].set_yscale('log')

axs[4, 0].set_ylabel(r'$\max(\left| \mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}\right|)$')

axs[0, 1].set_ylabel(
    r'$\frac{g_0\left[\mathbf{x}^{(k)}\right] - g_0\left[\mathbf{x}^{(k-1)}\right]}{\frac{\partial g_0^{(k-1)}}{\partial\mathbf{x}}\cdot \Delta \mathbf{x}^{(k-1)}}$')
axs[1, 1].set_ylabel(
    r'$\frac{g_0\left[\mathbf{x}^{(k)}\right] - g_0\left[\mathbf{x}^{(k-1)}\right]}{a}$')

axs[2, 1].set_ylabel(r'$\frac{4}{n} \sum_{i}^n \tilde{x}_i \left(1-\tilde{x}_i\right)$')

axs[3, 1].set_ylabel('N')

axs[4, 1].set_ylabel(
    r'$\frac{\Delta \mathbf{x}^{(k)} \cdot \Delta \mathbf{x}^{(k-1)}}{\left\|\Delta \mathbf{x}^{(k)} \right\| \cdot \left\|\Delta \mathbf{x}^{(k-1)}\right\|}$')

plt.show(block=True)
