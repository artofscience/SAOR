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
    c1=0
    c2=c1+1
    c3=c1+x
    c4=c2+x
    c5=c1+(y)*(x)
    c6=c2+(y)*(x)
    c7=c3+(y)*(x)
    c8=c4+(y)*(x)
    c=0
    for k in range(z-1):
        for j in range(y-1):
            for i in range(x-1):
#
                gon=vtk.vtkVoxel()
#               gon=vtk.vtkHexahedron() different ordering!!!
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
                c1=c1+1
                c2=c2+1
                c3=c3+1
                c4=c4+1
                c5=c5+1
                c6=c6+1
                c7=c7+1
                c8=c8+1
            c1=c1+1
            c2=c2+1
            c3=c3+1
            c4=c4+1
            c5=c5+1
            c6=c6+1
            c7=c7+1
            c8=c8+1
        c1=c1+x
        c2=c2+x
        c3=c3+x
        c4=c4+x
        c5=c5+x
        c6=c6+x
        c7=c7+x
        c8=c8+x
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
