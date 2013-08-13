import numpy as np
from BaseMesh import BaseMesh
from TensorMesh import TensorMesh
from DiffOperators import DiffOperators
from InnerProducts import InnerProducts
from utils import ndgrid, mkvc
import matplotlib.pyplot as plt




def linearInterp(coeffs, mesh, x):
    """
    Linear Interpolation

    given the coefficients on the cell-centers of the mesh (currently just a TensorMesh)

    compute the interpolation at the locations given in x (np.ndarray)

    data(x) := linearInterp(coeffs,mesh,x)

    """
    assert type(mesh) is TensorMesh, "Must be a TensorMesh"
    assert type(coeffs) is np.ndarray, "coeffs must be an np.ndarray"
    assert type(x) is np.ndarray, "x must be an np.ndarray"
    assert coeffs.size == mesh.nC, "coeffs must have the number of cells in the mesh"

    x = x.reshape(-1, mesh.dim, order='F')
    n = x.shape[0]
    datax = np.zeros(n)

    # pad coefficients to reduce number of cases
    coeffsP = np.zeros(mesh.n+2)


    ind = np.zeros((n, mesh.dim), dtype=int)

    # Must find where each x lies in our mesh
    # TODO: Interpolation: if this is regularly spaced you can be more efficient
    vecCC = [mesh.vectorCCx, mesh.vectorCCy, mesh.vectorCCz]
    h = [mesh.hx, mesh.hy, mesh.hz]
    for ii in range(n): # loop over all points x
        for d in range(mesh.dim):
            vecCCd = np.r_[vecCC[d][0]-h[d][0], vecCC[d], vecCC[d][-1]+h[d][-1] ]
            if (vecCCd[0] < x[ii,d]) and (x[ii,d] < vecCCd[-1]):
                # get index of left cell center
                dist = x[ii, d] - vecCCd
                ind[ii, d] = np.where(dist<0)[0][0]-1  # one before the dist vector turns negative.
            else:
                ind[ii, d] = -1000  # point is outside the valid domain
            print ind[ii, d]

    if mesh.dim==1:
        d = 0
        valid    = (ind[:,d]> -1000)
        indValid = ind[valid,d]
        print valid
        print indValid
        vecCCd = np.r_[vecCC[d][0]-h[d][0], vecCC[d], vecCC[d][-1]+h[d][-1] ]
        coeffsP[1:-1] = coeffs

        hfi  = (vecCCd [indValid+1] - vecCCd [indValid])
        wL = (vecCCd [indValid+1] -  x[valid, 0])/hfi
        wR = (x[valid, 0] - vecCCd [indValid]  )/hfi
        datax[valid] = wL *  coeffsP[indValid]  + wR * coeffsP[indValid+1]

        print 'datax', datax



    return datax






if __name__ == '__main__':

    hx = np.array([1,2,1])
    hy = np.array([1,2])
    hz = np.array([1,1,1,1])

    mesh = TensorMesh([hx])

    coeffs = np.array([0, 4, 3])
    x = np.linspace(-1, 5, 101)
    # x = np.array([-1,2,10])
    datax = linearInterp(coeffs, mesh, x)
    plt.plot(mesh.vectorCCx, coeffs, 'rx')
    plt.plot(x,datax,'b-')
    plt.show()
