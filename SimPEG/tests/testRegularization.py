import unittest
import sys
sys.path.append('../')
from TetraMesh import TetraMesh
import numpy as np
from GaussNewton import checkDerivative
from scipy import sparse as sp
from Regularization import elastic

class TestRegularizer2D(unittest.TestCase):

    def setUp(self):
        n = np.round(np.random.rand(2)*6)+1
        h1 = np.random.rand(n[0])
        h2 = np.random.rand(n[1])
        x0 = np.random.rand(2)

        self.mesh = TetraMesh([h1,h2],x0)


    def test_originElastic(self):
    	yRef = self.mesh.nodes.flatten(order='F')
    	y0   = yRef

    	Sc,dS,A   = elastic(self.mesh,y0,yRef,doDerivative=True)
    	self.assertTrue( np.abs(Sc)==0 and all(dS==0))

    def test_derivativeElastic(self):
    	yRef = self.mesh.nodes.flatten(order='F')
    	y0   = yRef + 1e-1*np.random.randn(self.mesh.nNodes*self.mesh.dim)

    	fctn = lambda y: elastic(self.mesh,y,yRef)
    	self.assertTrue(checkDerivative(fctn,y0,plotIt=False))

class TestRegularizer3D(unittest.TestCase):

    def setUp(self):
        n = np.round(np.random.rand(3)*6)+1
        h1 = np.random.rand(n[0])
        h2 = np.random.rand(n[1])
        h3 = np.random.rand(n[2])
        x0 = np.random.rand(3)

        self.mesh = TetraMesh([h1,h2,h3],x0)


    def test_originElastic(self):
    	yRef = self.mesh.nodes.flatten(order='F')
    	y0   = yRef

    	Sc,dS,A   = elastic(self.mesh,y0,yRef,doDerivative=True)
    	self.assertTrue( np.abs(Sc)==0 and all(dS==0))

   	def test_derivativeElastic(self):
   		yRef = self.mesh.nodes.flatten(order='F')
    	y0   = yRef + 1e-1*np.random.randn(self.mesh.nNodes*self.mesh.dim)

    	fctn = lambda y: elastic(self.mesh,y,yRef)
    	self.assertTrue(checkDerivative(fctn,y0,plotIt=False))

if __name__ == '__main__':
    unittest.main()