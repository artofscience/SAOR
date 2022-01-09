import matplotlib.pyplot as plt
from matplotlib import colors
from sao.problems import Subproblem
from sao.move_limits import AdaptiveMoveLimit, Bounds, MoveLimit
from sao.intervening_variables import Linear, MixedIntervening
from sao.intervening_variables.mma import MMA02 as MMA
from sao.intervening_variables.mma import MMA87A
from sao.approximations import Taylor1

from problems.topology_optimization.stress import StressCantilever
from examples.topopt.util.optimize import optimize


itercount = 50
nelx = 50
nely = 100

## SETUP SUBPROBLEMS

mma = Subproblem(Taylor1(MMA(sinit=0.2)), limits=[Bounds(0, 1)])
mma.set_name("MMA_asyinit_0.2")

mma_ml = Subproblem(Taylor1(MMA87A(t=1/3)), limits=[Bounds(0, 1)])
mma_ml.set_name("MMA_asyinit_0.2_ML_0.3")

lin_aml = Subproblem(Taylor1(Linear()), limits=[Bounds(0, 1), AdaptiveMoveLimit(0.1)])
lin_aml.set_name("LIN_AML_0.3")

mix_int = MixedIntervening(nelx * nely, 2, default=MMA(sinit=0.2))
mix_int.set_intervening(Linear(), resp=0)
mix_mma_lin = Subproblem(Taylor1(mix_int), limits=[Bounds(0, 1), MoveLimit(0.1)])
mix_mma_lin.set_name("MIX_MMA_asyinit0.2_LIN")

sub_problems = [mma_ml, lin_aml, mix_mma_lin]

figdes, axsdes = plt.subplots(len(sub_problems), sharex=True)

fig, axs = plt.subplots(5, 2, sharex=True)
x = range(0, itercount - 1)

for i, sub_problem in enumerate(sub_problems):
    problem = StressCantilever(nelx, nely)
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