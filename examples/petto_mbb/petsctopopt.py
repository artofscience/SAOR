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
        nex=120 +8*3*4
        ney=40  +8*4
        nez=40  +8*4
        self.n = nex*ney*nez
        self.m = 1
        self.crs = 6
        self.x0 = 0.12 * np.ones(self.n, dtype=float)
        self.x_min = 0e0 * np.ones_like(self.x0)
        self.x_max = 1e0 * np.ones_like(self.x0)
        self.name = 'PETSc TopOpt'
        self.x_opt = None
        self.f_opt = None
#
        ends=[".npy", ".flg", ".dat", ".info"]
        for file in os.listdir("./"):
            filename = os.fsdecode(file)
            for end in ends:
                if filename.endswith(end): 
                    os.system("rm "+filename)
#
        print("-"*80)
        print("Initialising PETScTopOpt ... ")
        print("-"*80)
        tmp="mpirun -n %d /media/dirkmunro/Terra/Code/topopt_in_petsc_solve_to_numpy/topopt"%self.crs
#
#       INCLUDE OTHER COMMAND LINE ARGUMENTS (to avoid petsc compilation for different cases)
#
        tmp=tmp+" -nx %d -ny %d -nz %d &"%(nex+1,ney+1,nez+1)
        os.system(tmp)
#
        while not os.path.exists("ready.flg"):
            time.sleep(1)
        print("-"*80)
        print("... ready. Continue with SAOr loop (first PETSc solve shows solver levels)...")
#
    def g(self, x):
#
        g = np.zeros((self.m + 1), dtype=float)
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
        for i in range(self.crs):
            tmp=np.load("df_%d.npy"%i)
            d=c+len(tmp)
            dgdx[0][c:d]=tmp
            c=c+len(tmp)
            d=d+len(tmp)
        c=0
        d=0
        for i in range(self.crs):
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
            print("-"*80)
#
    def setx(self,x):
#
        tmp="done.flg"
        if os.path.isfile(tmp):
            os.system("rm "+tmp)
        np.save("xxx.npy",x)
        print("-"*80)
#
if __name__ == "__main__":
#
    from problems.util.fd import finite_difference
#
    dx = 1e-7
 
