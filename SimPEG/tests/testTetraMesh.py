import unittest
import sys
sys.path.append('../')
from TetraMesh import TetraMesh
import numpy as np
from OrderTest import OrderTest
from GaussNewton import checkDerivative
from scipy import sparse as sp

class TestTetraMeshSetup2D(unittest.TestCase):

    def setUp(self):
        n = np.round(np.random.rand(2)*6)+1
        h1 = np.random.rand(n[0])
        h2 = np.random.rand(n[1])
        x0 = np.random.rand(2)

        self.mesh = TetraMesh([h1,h2],x0)
        self.n    = n
        self.x0   = x0
        self.h1   = h1
        self.h2   = h2

    def testMeshDimensions(self):
        self.assertTrue(self.mesh.dim, 2)

    def testnTetra(self):
        self.assertTrue(self.mesh.nTetra, 4*np.prod(self.n))

    def testnNodes(self):
        self.assertTrue(self.mesh.nNodes, np.prod(self.n+1)+np.prod(self.n))

    def testOrigin(self):
        self.assertTrue(all(self.mesh.x0 == self.x0))

    def testTetraListDim(self):
        self.assertTrue(self.mesh.tetra.shape == (self.mesh.nTetra,3))

    def testNodeListDim(self):
        self.assertTrue(self.mesh.nodes.shape == (self.mesh.nNodes,2))

    def testP1(self):
        P1 = self.mesh.P1*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,0],:]
        self.assertTrue(all( (P1 == Pt).flatten()))

    def testP2(self):
        P  = self.mesh.P2*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,1],:]
        self.assertTrue(all( (P == Pt).flatten()))

    def testP3(self):
        P  = self.mesh.P3*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,2],:]
        self.assertTrue(all( (P == Pt).flatten()))


    def testPC(self):
        PC = self.mesh.PC*self.mesh.nodes
        Pt = (1/3.0)*(self.mesh.P1+self.mesh.P2+self.mesh.P3)*self.mesh.nodes
        self.assertTrue(all( (PC ==  Pt).flatten()))



class TestTetraMeshProjectors2D(unittest.TestCase):

    def setUp(self):
        h1 = np.ones(1)
        h2 = np.ones(1)
        x0 = np.random.rand(2)

        self.mesh = TetraMesh([h1,h2],x0)
        self.x0   = x0
        self.h1   = h1
        self.h2   = h2


    def testBoundaryIndices(self):
        self.assertTrue(all( (self.mesh.boundaryIndices ==  np.r_[0,1,2,3]).flatten()))

    def testPE1(self):
        E1 = self.mesh.PE1*self.mesh.nodes
        Et = (np.r_[-1,-1,1,-1,1,1,-1,1]/2.0).reshape(4,2)
        self.assertTrue(np.linalg.norm(E1-Et)<1e-15)

    def testPE2(self):
        E1 = self.mesh.PE2*self.mesh.nodes
        Et = (np.r_[1,-1,1,1,-1,1,-1,-1]/2.0).reshape(4,2)
        self.assertTrue(np.linalg.norm(E1-Et)<1e-15)


class TestTetraMeshSetup3D(unittest.TestCase):

    def setUp(self):
        n = np.round(np.random.rand(3)*6)+1
        h1 = np.random.rand(n[0])
        h2 = np.random.rand(n[1])
        h3 = np.random.rand(n[2])
        x0 = np.random.rand(3)

        self.mesh = TetraMesh([h1,h2,h3],x0)
        self.n    = n
        self.x0   = x0
        self.h1   = h1
        self.h2   = h2
        self.h3   = h3

    def testMeshDimensions(self):
        self.assertTrue(self.mesh.dim, 3)

    def testnTetra(self):
        self.assertTrue(self.mesh.nTetra, 24*np.prod(self.n))

    def testnNodes(self):
        self.assertTrue(self.mesh.nNodes, np.prod(self.n+1) + np.prod(self.n) + np.prod(self.n+[1,0,0]) + np.prod(self.n+[0,1,0])+ np.prod(self.n+[0,0,1]))

    def testOrigin(self):
        self.assertTrue(all(self.mesh.x0 == self.x0))

    def testTetraListDim(self):
        self.assertTrue(self.mesh.tetra.shape == (self.mesh.nTetra,4))

    def testNodeListDim(self):
        self.assertTrue(self.mesh.nodes.shape == (self.mesh.nNodes,3))

    def testP1(self):
        P1 = self.mesh.P1*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,0],:]
        self.assertTrue(all( (P1 == Pt).flatten()))

    def testP2(self):
        P  = self.mesh.P2*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,1],:]
        self.assertTrue(all( (P == Pt).flatten()))

    def testP3(self):
        P  = self.mesh.P3*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,2],:]
        self.assertTrue(all( (P == Pt).flatten()))

    def testP4(self):
        P  = self.mesh.P4*self.mesh.nodes
        Pt = self.mesh.nodes[self.mesh.tetra[:,3],:]
        self.assertTrue(all( (P == Pt).flatten()))

    def testPC(self):
        PC = self.mesh.PC*self.mesh.nodes
        Pt = (1/4.0)*(self.mesh.P1+self.mesh.P2+self.mesh.P3+self.mesh.P4)*self.mesh.nodes
        self.assertTrue(all( (PC ==  Pt).flatten()))

class TestDiffOps(OrderTest):
    name = "Test Differential Operators"
    meshSizes = [8,16,32]
    meshTypes = ['uniformTetraMesh','randomTetraMesh']
    expectedOrders = 2.

    def getError(self):
        # Create some functions to integrate
        if self.meshDimension==2:
            fun = lambda x: np.sin(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])
            if self.testDim==1:
                sol = lambda x: 2*np.pi*np.cos(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])
                Di  = self.M.Dx
            elif self.testDim==2:
                sol = lambda x: 2*np.pi*np.sin(2*np.pi*x[:, 0])*np.cos(2*np.pi*x[:, 1])
                Di  = self.M.Dy
        elif self.meshDimension==3:
            fun = lambda x: np.sin(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])*np.sin(2*np.pi*x[:, 2])
            if self.testDim==1:
                sol = lambda x: 2*np.pi*np.cos(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])*np.sin(2*np.pi*x[:, 2])
                Di  = self.M.Dx
            elif self.testDim==2:
                sol = lambda x: 2*np.pi*np.sin(2*np.pi*x[:, 0])*np.cos(2*np.pi*x[:, 1])*np.sin(2*np.pi*x[:, 2])
                Di  = self.M.Dy
            elif self.testDim==3:
                sol = lambda x: 2*np.pi*np.sin(2*np.pi*x[:, 0])*np.sin(2*np.pi*x[:, 1])*np.cos(2*np.pi*x[:, 2])
                Di  = self.M.Dz

        sA = sol(self.M.PC*self.M.nodes)
        sN = Di*fun(self.M.nodes)
        err = np.linalg.norm(self.M.vol*(sA - sN), 2)
        return err

    def test_Dx(self):
        self.name = "Test Differential Operators - Dx in 3D"
        self.meshDimension = 3
        self.testDim = 1
        self.orderTest()

    def test_Dy(self):
        self.name = "Test Differential Operators - Dy in 3D"
        self.testDim = 2
        self.orderTest()

    def test_Dx2D(self):
        self.name = "Test Differential Operators - Dx in 2D"
        self.meshDimension = 2
        self.testDim = 1
        self.orderTest()

    def test_Dy2D(self):
        self.name = "Test Differential Operators - Dy in 2D"
        self.meshDimension = 2
        self.testDim = 2
        self.orderTest()

    def test_Dz(self):
        self.name = "Test Differential Operators - Dz in 3D"
        self.testDim = 3
        self.orderTest()

class TestDerivatives2D(unittest.TestCase):
    def setUp(self):
        n = np.round(np.random.rand(2)*6)+1
        h1 = np.random.rand(n[0])
        h2 = np.random.rand(n[1])
        x0 = np.random.rand(2)

        self.mesh = TetraMesh([h1,h2],x0)

    def test_volume(self):
        self.name = "TestDerivatives2D - volume"
        d = lambda y: self.mesh.getVolume(y)
        x0 = self.mesh.nodes.flatten(order='F')

        self.assertTrue(checkDerivative(d,x0,plotIt=False))

    def test_determinant(self):
        self.name = "TestDerivatives2D - determinant"
        d = lambda y: self.mesh.getDeterminant(y)
        x0 = self.mesh.nodes.flatten(order='F')

        self.assertTrue(checkDerivative(d,x0,plotIt=False))

class TestDerivatives3D(unittest.TestCase):
    def setUp(self):
        n = np.round(np.random.rand(3)*6)+1
        h1 = np.random.rand(n[0])
        h2 = np.random.rand(n[1])
        h3 = np.random.rand(n[2])
        x0 = np.random.rand(3)

        self.mesh = TetraMesh([h1,h2,h3],x0)

    def test_volume(self):
        self.name = "TestDerivatives3D - volume"
        d = lambda y: self.mesh.getVolume(y)
        x0 = self.mesh.nodes.flatten(order='F')

        self.assertTrue(checkDerivative(d,x0,plotIt=False))

    def test_determinant(self):
        self.name = "TestDerivatives3D - determinant"
        d = lambda y: self.mesh.getDeterminant(y)
        x0 = self.mesh.nodes.flatten(order='F')

        self.assertTrue(checkDerivative(d,x0,plotIt=False))

    def test_Cofactor(self):
        self.name = "TestDerivatives3D - cofactor"
        def fctn(y):
            cof,dCof = self.mesh.getCofactor(y)
            cof = np.r_[cof[:,0],cof[:,1],cof[:,2],cof[:,3],cof[:,4],cof[:,5],cof[:,6],cof[:,7],cof[:,8]]
            dCof = sp.vstack((dCof[0],dCof[1],dCof[2],dCof[3],dCof[4],dCof[5],dCof[6],dCof[7],dCof[8]))
            return cof,dCof

        x0 = self.mesh.nodes.flatten(order='F',plotIt=False)
        checkDerivative(fctn,x0)

if __name__ == '__main__':
    unittest.main()
