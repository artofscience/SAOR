#
import numpy as np
#
def finite_difference(prob, y, dx):
#
    x = y
    g0 = prob.g(x)
    dg_an = prob.dg(x)
#
    print(f"\nProblem:  {prob.name}")
    tmp=f"{' ':4}"
    for j in range(prob.m+1):
        tmp+=f"||F{j}{' ':28}"
    print(tmp)
#
    dg_fd = np.zeros_like(dg_an)
    err = np.zeros_like(dg_an)
    for i in range(prob.n):
        x0 = x[i]
        x[i] += dx
        gp = prob.g(x)
        x[i] = x0
        dg_fd[:, i] = (gp - g0) / dx
        err[:, i] = abs(dg_an[:, i] / dg_fd[:, i] - 1.0)
        if np.any(err>1e-6):
            print('WARNING:') 
        tmp=f"x_{i}{' ':1}||"
        for j in range(prob.m+1):
            tmp+=f"{dg_an[j, i]:9.1e}|"
            tmp+=f"{dg_fd[j, i]:9.1e}|"
            tmp+=f"{err[j,i]:9.1e} ||"
        print(tmp)
#
