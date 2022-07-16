#
import os
import numpy as np
import time
from sao.problems.problem import Problem
#
class PETScTopOpt(Problem):
#
    def __init__(self):
        super().__init__()
        self.n = (136+16)*(72+16)*(72+16)
        self.m = 1
        self.cores = 6
        self.x0 = 0.12 * np.ones(self.n, dtype=float)
        self.x_min = 0e0 * np.ones_like(self.x0)
        self.x_max = 1e0 * np.ones_like(self.x0)
        self.name = 'PETSc TopOpt'
        self.x_opt = None
        self.f_opt = None
#
        os.system("rm *.npy > log.log")
        os.system("rm *.flg > log.log")
        os.system("rm RestartSol00.dat > log.log")
        os.system("rm RestartSol00.dat.info > log.log")
#
#       tmp="/media/dirkmunro/Terra/Code/topopt_in_petsc_solve_to_numpy/topopt -restartFileVecSol RestartSol00.dat &"
        tmp="mpirun -n %d /media/dirkmunro/Terra/Code/topopt_in_petsc_solve_to_numpy/topopt -restartFileVecSol RestartSol00.dat &"\
            %self.cores
        os.system(tmp)
#
        while not os.path.exists("ready.flg"):
            time.sleep(1)
        print("TopOpt Petsc is ready")
#
    def g(self, x):
#
        g = np.zeros((self.m + 1), dtype=float)
#       self.setx(x)
#       self.solve()
        g[:]=self.read_g()
#
        return g

    def dg(self, x):
#
        start = time.time()
        dg = np.zeros((self.m + 1, self.n), dtype=float)
        dg[:]=self.read_dg()
        end = time.time()
        return dg
#
    def read_g(self):
#
        tmp=np.load('fg.npy')
        return tmp
#
    def read_dg(self):
#
        dgdx=np.zeros((self.m+1,self.n), dtype=float)
#
        c=0
        d=0
        for i in range(self.cores):
            tmp=np.load("df_%d.npy"%i)
            d=c+len(tmp)
            dgdx[0][c:d]=tmp
            c=c+len(tmp)
            d=d+len(tmp)
        c=0
        d=0
        for i in range(self.cores):
            tmp=np.load("dg_%d.npy"%i)
            d=c+len(tmp)
            dgdx[1][c:d]=tmp
            c=c+len(tmp)
            d=d+len(tmp)
#
        return dgdx
#
    def solve(self,k):
#
        open("gowith_%d.flg"%k, 'w').close()
#
        if not os.path.exists("exit.flg"):
            while not os.path.exists("done.flg"):
                time.sleep(1)
                time.sleep(1)
            print("Topopt petsc is done done....")
#
    def setx(self,x):
#
        os.system("rm done.flg > log.log")
        np.save("xxx.npy",x)
#
if __name__ == "__main__":
#
    from problems.util.fd import finite_difference
#
