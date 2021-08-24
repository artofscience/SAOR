import numpy as np
import logging
from Problems.topology_optimization.compliance import Compliance
from sao.solvers.optimality_criteria import oc
from sao.convergence_criteria.criteria import VariableChange
from sao.util import Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
np.set_printoptions(precision=4)


def example_compliance_mbb(nelx=100, nely=50, volfrac=0.4):
    logger.info("Solving compliance minimization subject to volume constraint")

    # Instantiate problem
    prob = Compliance(nelx, nely, volfrac, penal=3, rmin=3)

    # Instantiate solver
    x_k = prob.x0.copy()
    vis = None
    counter = 0
    converged = VariableChange(x_k)

    plotter = Plot(['objective'], path=".")
    # Optimization loop
    a = 0
    while not converged:
        f = prob.g(x_k)
        df = prob.dg(x_k)
        x_k[:], a = oc(x_k, df, gt=a)
        print(a)

        # Print & Plot
        vis = prob.visualize(x_k, counter, vis)
        logger.info('counter: {:^4d}  |  obj: {:^9.3f}  |  constr: {:^6.3f}  |  volume: {:^6.3f}'.format(
            counter, f[0], f[1], np.mean(np.asarray(prob.H * x_k[np.newaxis].T / prob.Hs)[:, 0])))
        plotter.plot([f[0]])
        counter += 1

    logger.info('Optimization loop converged!')


if __name__ == "__main__":
    example_compliance_mbb(nelx=100, nely=50, volfrac=0.5)
