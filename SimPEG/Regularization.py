import numpy as np
from scipy import sparse as sp
from utils import  mkvc,sdiag,speye,spzeros
from TetraMesh import TetraMesh
from GaussNewton import checkDerivative

def hyperelastic(Mesh,yc,yRef,doDerivative=True,alpha=1,alphaLength=1,alphaArea=1,alphaVolume=1):
	"""
		hyperelastic regularizer for image registration
	"""

	pass


def elastic(Mesh,yc,yRef=None,doDerivative=True,alpha=1,lamb=0,mu=1):
	"""
		linear elastic regularizer for tetrahedral finite element discretization
		with piecewise linear basis functions

		S(u) = \alpha \int || B * u  ||^2 dx

	 	where B is the navier lame operator:

	 			| a*dx1			|
	 			| a*dx2 		|
	 		B=	|		a*dx1	|  , (d=2) and a=sqrt(mu) and b=sqrt(mu+lamb)
	 			|		a*dx2	|
	 			| b*dx1 b*dx2	|


	 	or

	 			| a*dx1					|
	 			| a*dx2 				|
	 			| a*dx3 				|
	 		   	|		a*dx1			|  , (d=3) and a=sqrt(mu) and b=sqrt(mu+lamb)
	 			|		a*dx2			|
	 		B=	|		a*dx3			|
	 			|				a*dx1	|
	 			|				a*dx2	|
	 			|				a*dx3	|
	 			| b*dx1 b*dx2	b*dx3	|


	Input:
	  	Mesh  - TetraMesh
	  	uc    - coefficients of current deformation
	  	yRef  - coefficients of reference transformation, default: yRef = Mesh.nodes

	Optional Input
		doDerivative 	- flag to compute derivatives, default: True
		alpha 			- default: 1
		mu    			- default: 1
		lamb 			- default:0

	"""

	# check if yc is valid
	assert type(yc) == np.ndarray, 		"yc must be a numpy array."
	assert yc.size == Mesh.nodes.size, 	"yc is of incorrect size."

   	if yRef is None:
   		yRef = Mesh.nodes.flatten(order='F')
   	else:
	 	assert type(yRef) == np.ndarray, 		"yRef must be a numpy array."
	 	assert yRef.size == Mesh.nodes.size, 	"yRef is of incorrect size."
	 	yRef = yRef.flatten(order='F')

	uc = mkvc(yc-yRef,1)

	a = np.sqrt(mu)
	b = np.sqrt(lamb)

	# build Hessian
	B = sp.vstack((sp.kron(a*speye(Mesh.dim), Mesh.GRAD), b*Mesh.DIV))
	V = sp.kron(speye(Mesh.dim**2+1), sdiag(Mesh.vol))
	A = B.T.dot(V.dot(B))

	dS = A.dot(uc)
	Sc = 0.5*(np.dot(uc.T, dS))*np.ones(1)

	if not(doDerivative):
		return Sc
	else:
		return Sc,dS,A

if __name__ == '__main__':
    print('Welcome to Regularization!')

    testDim = 2
    h1 = np.ones(1)
    h2 = np.ones(1)
    h3 = np.ones(1)
    x0 = np.zeros((3, 1))

    if testDim == 1:
        h = [h1]
        x0 = x0[0]
    elif testDim == 2:
        h = [h1, h2]
        x0 = x0[0:2]
    else:
        h = [h1, h2, h3]

    I = np.linspace(0, 1, 8)
    M = TetraMesh(h, x0)


    yRef = M.nodes.flatten(order='F')
    yc   = yRef + 1e-1*np.random.randn(M.nNodes*M.dim)

    fctn = lambda y: elastic(M,y,yRef)
    checkDerivative(fctn,yc)
    Sc = elastic(M,yc,yRef)
    print Sc



