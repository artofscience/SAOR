import vtk
from vtk import *
import numpy as np
#
if __name__ == "__main__":
#
    pts=vtk.vtkPoints()
    cel=vtk.vtkCellArray()
    ply=vtk.vtkPolyData()
    uns=vtk.vtkUnstructuredGrid()
#
    rho=vtk.vtkFloatArray()
#
    x=65
    y=33
    z=33
    pts.SetNumberOfPoints(int(x*y*z))
#
    c=0
    for k in range(z):
        for j in range(y):
            for i in range(x):
                pts.SetPoint(c,float(i),float(j),float(k))
                c=c+1

#
    d=np.load('xxx.npy')
#
    print(d,max(d),min(d))
#
    c=0
#
    e_x=x-1
    e_y=y-1
    e_z=z-1
    n_x=x
    n_y=y
    n_z=z
#
    for k in range(e_z):
        for j in range(e_y):
            for i in range(e_x):
#
                c1 = i + j*n_x + k*n_x*n_y
                c2 = i + 1 + j*n_x + k*n_x*n_y
                c3 = i + 1 + (j+1)*n_x + k*n_x*n_y
                c4 = i + (j+1)*n_x + k*n_x*n_y

                c5 = i + j*n_x + (k+1)*n_x*n_y
                c6 = i+1 + (j)*n_x + (k+1)*n_x*n_y
                c7 = i+1 + (j+1)*n_x + (k+1)*n_x*n_y
                c8 = i + (j+1)*n_x + (k+1)*n_x*n_y
#
#               print(c1,c2,c3,c4,c5,c6,c7,c8)
#
#               gon=vtk.vtkVoxel()
                gon=vtk.vtkHexahedron() #different ordering!!!
                gon.GetPointIds().SetId(0,c1)
                gon.GetPointIds().SetId(1,c2)
                gon.GetPointIds().SetId(2,c3)
                gon.GetPointIds().SetId(3,c4)
                gon.GetPointIds().SetId(4,c5)
                gon.GetPointIds().SetId(5,c6)
                gon.GetPointIds().SetId(6,c7)
                gon.GetPointIds().SetId(7,c8)
                ids=uns.InsertNextCell(gon.GetCellType(),gon.GetPointIds())
#
                rho.InsertTuple1(c,d[c])
#
                c=c+1
#
    uns.SetPoints(pts)
#
    uns.GetCellData().AddArray(rho)
#
    writer = vtk.vtkXMLUnstructuredGridWriter()
#   writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName('lattice.vtu')
    writer.SetInputData(uns)
    writer.Write()
#

#
