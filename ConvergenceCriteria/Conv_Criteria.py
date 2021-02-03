## Imports
import numpy as np
import constants as ct


## Contains all the possible convergence criteria one might use
class ConvergenceCriteria:

    ## Constructor of class
    def __init__(self, criterion_name, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.name = criterion_name
        self.converged = False          # When initialized, criterion is not satisfied (to enter optimization loop)

    ## Function to calculate |KKT|: Filter out bound constraints so that KKT goes to 0 when convergence is achieved
    def get_KKT_norm(self, x, dg_j, lam):
        KKT_red_idx = [index for index, value in enumerate(x)
                       if 1.001 * self.xmin[index] < value < 0.99 * self.xmax[index]]
        KKT_reduced = dg_j[0, KKT_red_idx] + np.dot(lam.T, dg_j[1:, KKT_red_idx])
        KKT_res = np.linalg.norm(KKT_reduced)
        return KKT_res

    ## Function to calculate the Normalized Change in Design Variables
    def get_VarChange(self, x_k, xold1):
        VarChange_norm = np.linalg.norm((x_k - xold1) / (self.xmax - self.xmin))
        return VarChange_norm

    ## Function to calculate the Normalized Change in Objective Function Values
    @staticmethod
    def get_ObjChange(g_0, g_0old1):
        if abs(g_0old1) > 1e-5:
            ObjChange_abs = abs((g_0 - g_0old1) / g_0old1)
        else:
            ObjChange_abs = abs((g_0 - g_0old1))
        return ObjChange_abs

    ## Function to calculate if the convergence criterion is satisfied at the current iteration
    def get_Convergence(self, **kwargs):
        x_k = kwargs.get('design', None)
        g = kwargs.get('responses', None)
        gold1 = kwargs.get('old_responses', None)
        dg = kwargs.get('sensitivities', None)
        lam = kwargs.get('lagrange_multipliers', None)
        xold1 = kwargs.get('xold1', None)
        if self.name == 'KKT':
            KKT_res = self.get_KKT_norm(x_k, dg, lam)
            if KKT_res < ct.TOLERANCE:
                self.converged = True
        elif self.name == 'VariableChange':
            VarChange_norm = self.get_VarChange(x_k, xold1)
            if (VarChange_norm < ct.TOLERANCE) and (np.all(g[1:] < ct.TOLERANCE)):
                self.converged = True
        elif self.name == 'ObjectiveChange':
            ObjChange_abs = self.get_ObjChange(g[0], gold1[0])
            if (ObjChange_abs < ct.TOLERANCE) and (np.all(g[1:] < ct.TOLERANCE)):
                self.converged = True
        elif self.name == 'AllTogether':
            KKT_res = self.get_KKT_norm(x_k, dg, lam)
            VarChange_norm = self.get_VarChange(x_k, xold1)
            ObjChange_abs = self.get_ObjChange(g[0], gold1[0])
            if (KKT_res < ct.TOLERANCE) and (VarChange_norm < ct.TOLERANCE) and (ObjChange_abs < ct.TOLERANCE) and \
               (np.all(g[1:] < ct.TOLERANCE)):
                self.converged = True
