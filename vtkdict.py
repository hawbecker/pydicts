import sys
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib.pyplot as plt
from matplotlib import cm
import subprocess
from matplotlib.colors import Normalize as Normalize

class vtkdata():
    '''
    Class to store data from vtk (e.g. UAvg, uu', etc)
    '''
    def __init__(self, file, var):
        '''
        file: which vtk file to read
        var: which variable to store in self.data
        '''
        self.file = file
        self.var = var
        self.readvtk() # load nloc, c2n, cell data
#        self.calc_cc() # calculate cell center locations

    def readvtk(self):
        '''
        loads nloc, c2n, cell data into numpy arrays
        '''
        # load VTK file data into Numpy arrays
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.file)
        reader.Update()
        output = reader.GetOutput()
        self.NN   = output.GetNumberOfPoints()                                  # number of nodes
        self.cells = output.GetPolys()
        self.triangles = self.cells.GetData()
        self.NC   = output.GetNumberOfCells()                                   # number of cells
        self.grid = vtk_to_numpy(output.GetPoints().GetData())                  # node locations: x, y, z
        if self.NN < self.NC: # I'm not sure this is the best way to determine which to use... may need better way!!!
            self.nvls = output.GetPointData().GetArray(self.var).GetNumberOfTuples()
            self.data = output.GetPointData().GetArray(self.var)       # data values for each cell
            self.x   = np.zeros((self.NN))
            self.y   = np.zeros((self.NN))
            for i in xrange(self.NN):
                pt = output.GetPoint(i)
                self.x[i] = pt[0]/1000.0
                self.y[i] = pt[1]/1000.0
        else:
            self.data = output.GetCellData().GetArray(self.var)       # data values for each cell
            self.nvls = output.GetCellData().GetArray(self.var).GetNumberOfTuples()
            self.x   = np.zeros((self.NC))
            self.y   = np.zeros((self.NC))
            for i in xrange(self.NC):
                pt = output.GetPoint(i)
                self.x[i] = pt[0]/1000.0
                self.y[i] = pt[1]/1000.0
        self.ntri = output.GetPolys().GetData().GetNumberOfTuples()/4
        self.tri = np.zeros((self.ntri,3))
        self.ux  = np.zeros((self.nvls))
        self.uy  = np.zeros((self.nvls))
        self.uz  = np.zeros((self.nvls))
        for i in xrange(0,self.ntri):
            self.tri[i, 0] = self.triangles.GetTuple(4*i + 1)[0]
            self.tri[i, 1] = self.triangles.GetTuple(4*i + 2)[0]
            self.tri[i, 2] = self.triangles.GetTuple(4*i + 3)[0]
        for i in xrange(0,self.data.GetNumberOfTuples()):
            U = self.data.GetTuple(i)
            self.ux[i] = U[0]
            self.uy[i] = U[1]
            self.uz[i] = U[2]

    def calc_cc(self):
        '''
        calculates cell center location for each cell
        '''
        self.cc = np.zeros((self.NC, 3), dtype=float)
        for i in range(0, self.NC):
            self.cc[i,0] = np.mean(self.grid[self.c2n[i,:], 0])                 # cell center x
            self.cc[i,1] = np.mean(self.grid[self.c2n[i,:], 1])                 # cell center y
            self.cc[i,2] = np.mean(self.grid[self.c2n[i,:], 2])                 # cell center z

