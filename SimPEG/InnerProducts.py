from scipy import sparse as sp
from sputils import sdiag, inv3X3BlockDiagonal, inv2X2BlockDiagonal
from utils import sub2ind, ndgrid, mkvc, getSubArray
import numpy as np


class InnerProducts(object):
    """
        Class creates the inner product matrices that you need!
    """
    def __init__(self):
        raise Exception('InnerProducts is a base class providing inner product matrices for meshes and cannot run on its own. Inherit to your favorite Mesh class.')

    def getFaceInnerProduct(self, mu=None, returnP=False):
        if self.dim == 2:
            return getFaceInnerProduct2D(self, mu, returnP)
        elif self.dim == 3:
            return getFaceInnerProduct(self, mu, returnP)

    def getEdgeInnerProduct(self, sigma=None, returnP=False):
        if self.dim == 2:
            return getEdgeInnerProduct2D(self, sigma, returnP)
        elif self.dim == 3:
            return getEdgeInnerProduct(self, sigma, returnP)

# ------------------------ Geometries ------------------------------
#
#
#         node(i,j,k+1) ------ edge2(i,j,k+1) ----- node(i,j+1,k+1)
#              /                                    /
#             /                                    / |
#         edge3(i,j,k)     face1(i,j,k)        edge3(i,j+1,k)
#           /                                    /   |
#          /                                    /    |
#    node(i,j,k) ------ edge2(i,j,k) ----- node(i,j+1,k)
#         |                                     |    |
#         |                                     |   node(i+1,j+1,k+1)
#         |                                     |    /
#    edge1(i,j,k)      face3(i,j,k)        edge1(i,j+1,k)
#         |                                     |  /
#         |                                     | /
#         |                                     |/
#    node(i+1,j,k) ------ edge2(i+1,j,k) ----- node(i+1,j+1,k)


def getFaceInnerProduct(mesh, mu=None, returnP=False):

    if mu is None:  # default is ones
        mu = np.ones((mesh.nC, 1))

    m = np.array([mesh.nCx, mesh.nCy, mesh.nCz])
    nc = mesh.nC

    i, j, k = np.int64(range(m[0])), np.int64(range(m[1])), np.int64(range(m[2]))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    if mesh._meshType == 'LOM':
        fN1 = mesh.r(mesh.normals, 'F', 'Fx', 'M')
        fN2 = mesh.r(mesh.normals, 'F', 'Fy', 'M')
        fN3 = mesh.r(mesh.normals, 'F', 'Fz', 'M')

    def Pxxx(pos):
        ind1 = sub2ind(mesh.nFx, np.c_[ii + pos[0][0], jj + pos[0][1], kk + pos[0][2]])
        ind2 = sub2ind(mesh.nFy, np.c_[ii + pos[1][0], jj + pos[1][1], kk + pos[1][2]]) + mesh.nF[0]
        ind3 = sub2ind(mesh.nFz, np.c_[ii + pos[2][0], jj + pos[2][1], kk + pos[2][2]]) + mesh.nF[0] + mesh.nF[1]

        IND = np.r_[ind1, ind2, ind3].flatten()

        PXXX = sp.coo_matrix((np.ones(3*nc), (range(3*nc), IND)), shape=(3*nc, np.sum(mesh.nF))).tocsr()

        if mesh._meshType == 'LOM':
            I3x3 = inv3X3BlockDiagonal(getSubArray(fN1[0], [i + pos[0][0], j + pos[0][1], k + pos[0][2]]), getSubArray(fN1[1], [i + pos[0][0], j + pos[0][1], k + pos[0][2]]), getSubArray(fN1[2], [i + pos[0][0], j + pos[0][1], k + pos[0][2]]),
                                       getSubArray(fN2[0], [i + pos[1][0], j + pos[1][1], k + pos[1][2]]), getSubArray(fN2[1], [i + pos[1][0], j + pos[1][1], k + pos[1][2]]), getSubArray(fN2[2], [i + pos[1][0], j + pos[1][1], k + pos[1][2]]),
                                       getSubArray(fN3[0], [i + pos[2][0], j + pos[2][1], k + pos[2][2]]), getSubArray(fN3[1], [i + pos[2][0], j + pos[2][1], k + pos[2][2]]), getSubArray(fN3[2], [i + pos[2][0], j + pos[2][1], k + pos[2][2]]))
            PXXX = I3x3 * PXXX

        return PXXX

    # no  | node        | f1        | f2        | f3
    # 000 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k
    # 100 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k
    # 010 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k
    # 110 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k
    # 001 | i  ,j  ,k   | i  , j, k | i, j  , k | i, j, k+1
    # 101 | i+1,j  ,k   | i+1, j, k | i, j  , k | i, j, k+1
    # 011 | i  ,j+1,k   | i  , j, k | i, j+1, k | i, j, k+1
    # 111 | i+1,j+1,k   | i+1, j, k | i, j+1, k | i, j, k+1

    # Square root of cell volume multiplied by 1/8
    v = np.sqrt(0.125*mesh.vol)
    V3 = sdiag(np.r_[v, v, v])  # We will multiply on each side to keep symmetry

    P000 = V3*Pxxx([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    P100 = V3*Pxxx([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    P010 = V3*Pxxx([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    P110 = V3*Pxxx([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    P001 = V3*Pxxx([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    P101 = V3*Pxxx([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    P011 = V3*Pxxx([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    P111 = V3*Pxxx([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if mu.size == mesh.nC:  # Isotropic!
        mu = mkvc(mu)  # ensure it is a vector.
        Mu = sdiag(np.r_[mu, mu, mu])
    elif mu.shape[1] == 3:  # Diagonal tensor
        Mu = sdiag(np.r_[mu[:, 0], mu[:, 1], mu[:, 2]])
    elif mu.shape[1] == 6:  # Fully anisotropic
        row1 = sp.hstack((sdiag(mu[:, 0]), sdiag(mu[:, 3]), sdiag(mu[:, 4])))
        row2 = sp.hstack((sdiag(mu[:, 3]), sdiag(mu[:, 1]), sdiag(mu[:, 5])))
        row3 = sp.hstack((sdiag(mu[:, 4]), sdiag(mu[:, 5]), sdiag(mu[:, 2])))
        Mu = sp.vstack((row1, row2, row3))

    A = P000.T*Mu*P000 + P001.T*Mu*P001 + P010.T*Mu*P010 + P011.T*Mu*P011 + P100.T*Mu*P100 + P101.T*Mu*P101 + P110.T*Mu*P110 + P111.T*Mu*P111
    P = [P000, P001, P010, P011, P100, P101, P110, P111]
    if returnP:
        return A, P
    else:
        return A


def getFaceInnerProduct2D(mesh, mu=None, returnP=False):

    if mu is None:  # default is ones
        mu = np.ones((mesh.nC, 1))

    m = np.array([mesh.nCx, mesh.nCy])
    nc = mesh.nC

    i, j = np.int64(range(m[0])), np.int64(range(m[1]))

    iijj = ndgrid(i, j)
    ii, jj = iijj[:, 0], iijj[:, 1]

    if mesh._meshType == 'LOM':
        fN1 = mesh.r(mesh.normals, 'F', 'Fx', 'M')
        fN2 = mesh.r(mesh.normals, 'F', 'Fy', 'M')

    def Pxx(pos):
        ind1 = sub2ind(mesh.nFx, np.c_[ii + pos[0][0], jj + pos[0][1]])
        ind2 = sub2ind(mesh.nFy, np.c_[ii + pos[1][0], jj + pos[1][1]]) + mesh.nF[0]

        IND = np.r_[ind1, ind2].flatten()

        PXX = sp.coo_matrix((np.ones(2*nc), (range(2*nc), IND)), shape=(2*nc, np.sum(mesh.nF))).tocsr()

        if mesh._meshType == 'LOM':
            I2x2 = inv2X2BlockDiagonal(getSubArray(fN1[0], [i + pos[0][0], j + pos[0][1]]), getSubArray(fN1[1], [i + pos[0][0], j + pos[0][1]]),
                                       getSubArray(fN2[0], [i + pos[1][0], j + pos[1][1]]), getSubArray(fN2[1], [i + pos[1][0], j + pos[1][1]]))
            PXX = I2x2 * PXX

        return PXX

    # no | node      | f1     | f2
    # 00 | i  ,j     | i  , j | i, j
    # 10 | i+1,j     | i+1, j | i, j
    # 01 | i  ,j+1   | i  , j | i, j+1
    # 11 | i+1,j+1   | i+1, j | i, j+1

    # Square root of cell volume multiplied by 1/4
    v = np.sqrt(0.25*mesh.vol)
    V2 = sdiag(np.r_[v, v])  # We will multiply on each side to keep symmetry

    P00 = V2*Pxx([[0, 0], [0, 0]])
    P10 = V2*Pxx([[1, 0], [0, 0]])
    P01 = V2*Pxx([[0, 0], [0, 1]])
    P11 = V2*Pxx([[1, 0], [0, 1]])

    if mu.size == mesh.nC:  # Isotropic!
        mu = mkvc(mu)  # ensure it is a vector.
        Mu = sdiag(np.r_[mu, mu])
    elif mu.shape[1] == 2:  # Diagonal tensor
        Mu = sdiag(np.r_[mu[:, 0], mu[:, 1]])
    elif mu.shape[1] == 3:  # Fully anisotropic
        row1 = sp.hstack((sdiag(mu[:, 0]), sdiag(mu[:, 2])))
        row2 = sp.hstack((sdiag(mu[:, 2]), sdiag(mu[:, 1])))
        Mu = sp.vstack((row1, row2))

    A = P00.T*Mu*P00 + P10.T*Mu*P10 + P01.T*Mu*P01 + P11.T*Mu*P11
    P = [P00, P10, P01, P11]
    if returnP:
        return A, P
    else:
        return A


def getEdgeInnerProduct(mesh, sigma=None, returnP=False):

    if sigma is None:  # default is ones
        sigma = np.ones((mesh.nC, 1))

    m = np.array([mesh.nCx, mesh.nCy, mesh.nCz])
    nc = mesh.nC

    i, j, k = np.int64(range(m[0])), np.int64(range(m[1])), np.int64(range(m[2]))

    iijjkk = ndgrid(i, j, k)
    ii, jj, kk = iijjkk[:, 0], iijjkk[:, 1], iijjkk[:, 2]

    if mesh._meshType == 'LOM':
        eT1 = mesh.r(mesh.tangents, 'E', 'Ex', 'M')
        eT2 = mesh.r(mesh.tangents, 'E', 'Ey', 'M')
        eT3 = mesh.r(mesh.tangents, 'E', 'Ez', 'M')

    def Pxxx(pos):
        ind1 = sub2ind(mesh.nEx, np.c_[ii + pos[0][0], jj + pos[0][1], kk + pos[0][2]])
        ind2 = sub2ind(mesh.nEy, np.c_[ii + pos[1][0], jj + pos[1][1], kk + pos[1][2]]) + mesh.nE[0]
        ind3 = sub2ind(mesh.nEz, np.c_[ii + pos[2][0], jj + pos[2][1], kk + pos[2][2]]) + mesh.nE[0] + mesh.nE[1]

        IND = np.r_[ind1, ind2, ind3].flatten()

        PXXX = sp.coo_matrix((np.ones(3*nc), (range(3*nc), IND)), shape=(3*nc, np.sum(mesh.nE))).tocsr()

        if mesh._meshType == 'LOM':
            I3x3 = inv3X3BlockDiagonal(getSubArray(eT1[0], [i + pos[0][0], j + pos[0][1], k + pos[0][2]]), getSubArray(eT1[1], [i + pos[0][0], j + pos[0][1], k + pos[0][2]]), getSubArray(eT1[2], [i + pos[0][0], j + pos[0][1], k + pos[0][2]]),
                                       getSubArray(eT2[0], [i + pos[1][0], j + pos[1][1], k + pos[1][2]]), getSubArray(eT2[1], [i + pos[1][0], j + pos[1][1], k + pos[1][2]]), getSubArray(eT2[2], [i + pos[1][0], j + pos[1][1], k + pos[1][2]]),
                                       getSubArray(eT3[0], [i + pos[2][0], j + pos[2][1], k + pos[2][2]]), getSubArray(eT3[1], [i + pos[2][0], j + pos[2][1], k + pos[2][2]]), getSubArray(eT3[2], [i + pos[2][0], j + pos[2][1], k + pos[2][2]]))
            PXXX = I3x3 * PXXX

        return PXXX

    # no  | node        | e1          | e2          | e3
    # 000 | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k   | i  ,j  ,k
    # 100 | i+1,j  ,k   | i  ,j  ,k   | i+1,j  ,k   | i+1,j  ,k
    # 010 | i  ,j+1,k   | i  ,j+1,k   | i  ,j  ,k   | i  ,j+1,k
    # 110 | i+1,j+1,k   | i  ,j+1,k   | i+1,j  ,k   | i+1,j+1,k
    # 001 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k+1 | i  ,j  ,k
    # 101 | i+1,j  ,k+1 | i  ,j  ,k+1 | i+1,j  ,k+1 | i+1,j  ,k
    # 011 | i  ,j+1,k+1 | i  ,j+1,k+1 | i  ,j  ,k+1 | i  ,j+1,k
    # 111 | i+1,j+1,k+1 | i  ,j+1,k+1 | i+1,j  ,k+1 | i+1,j+1,k

    # Square root of cell volume multiplied by 1/8
    v = np.sqrt(0.125*mesh.vol)
    V3 = sdiag(np.r_[v, v, v])  # We will multiply on each side to keep symmetry

    P000 = V3*Pxxx([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    P100 = V3*Pxxx([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    P010 = V3*Pxxx([[0, 1, 0], [0, 0, 0], [0, 1, 0]])
    P110 = V3*Pxxx([[0, 1, 0], [1, 0, 0], [1, 1, 0]])
    P001 = V3*Pxxx([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    P101 = V3*Pxxx([[0, 0, 1], [1, 0, 1], [1, 0, 0]])
    P011 = V3*Pxxx([[0, 1, 1], [0, 0, 1], [0, 1, 0]])
    P111 = V3*Pxxx([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

    if sigma.size == mesh.nC:  # Isotropic!
        sigma = mkvc(sigma)  # ensure it is a vector.
        Sigma = sdiag(np.r_[sigma, sigma, sigma])
    elif sigma.shape[1] == 3:  # Diagonal tensor
        Sigma = sdiag(np.r_[sigma[:, 0], sigma[:, 1], sigma[:, 2]])
    elif sigma.shape[1] == 6:  # Fully anisotropic
        row1 = sp.hstack((sdiag(sigma[:, 0]), sdiag(sigma[:, 3]), sdiag(sigma[:, 4])))
        row2 = sp.hstack((sdiag(sigma[:, 3]), sdiag(sigma[:, 1]), sdiag(sigma[:, 5])))
        row3 = sp.hstack((sdiag(sigma[:, 4]), sdiag(sigma[:, 5]), sdiag(sigma[:, 2])))
        Sigma = sp.vstack((row1, row2, row3))

    A = P000.T*Sigma*P000 + P001.T*Sigma*P001 + P010.T*Sigma*P010 + P011.T*Sigma*P011 + P100.T*Sigma*P100 + P101.T*Sigma*P101 + P110.T*Sigma*P110 + P111.T*Sigma*P111
    P = [P000, P001, P010, P011, P100, P101, P110, P111]
    if returnP:
        return A, P
    else:
        return A


def getEdgeInnerProduct2D(mesh, sigma=None, returnP=False):

    if sigma is None:  # default is ones
        sigma = np.ones((mesh.nC, 1))

    m = np.array([mesh.nCx, mesh.nCy])
    nc = mesh.nC

    i, j = np.int64(range(m[0])), np.int64(range(m[1]))

    iijj = ndgrid(i, j)
    ii, jj = iijj[:, 0], iijj[:, 1]

    if mesh._meshType == 'LOM':
        eT1 = mesh.r(mesh.tangents, 'E', 'Ex', 'M')
        eT2 = mesh.r(mesh.tangents, 'E', 'Ey', 'M')

    def Pxx(pos):
        ind1 = sub2ind(mesh.nEx, np.c_[ii + pos[0][0], jj + pos[0][1]])
        ind2 = sub2ind(mesh.nEy, np.c_[ii + pos[1][0], jj + pos[1][1]]) + mesh.nE[0]

        IND = np.r_[ind1, ind2].flatten()

        PXX = sp.coo_matrix((np.ones(2*nc), (range(2*nc), IND)), shape=(2*nc, np.sum(mesh.nE))).tocsr()

        if mesh._meshType == 'LOM':
            I2x2 = inv2X2BlockDiagonal(getSubArray(eT1[0], [i + pos[0][0], j + pos[0][1]]), getSubArray(eT1[1], [i + pos[0][0], j + pos[0][1]]),
                                       getSubArray(eT2[0], [i + pos[1][0], j + pos[1][1]]), getSubArray(eT2[1], [i + pos[1][0], j + pos[1][1]]))
            PXX = I2x2 * PXX

        return PXX

    # no | node      | e1      | e2
    # 00 | i  ,j     | i  ,j   | i  ,j
    # 10 | i+1,j     | i  ,j   | i+1,j
    # 01 | i  ,j+1   | i  ,j+1 | i  ,j
    # 11 | i+1,j+1   | i  ,j+1 | i+1,j

    # Square root of cell volume multiplied by 1/4
    v = np.sqrt(0.25*mesh.vol)
    V2 = sdiag(np.r_[v, v])  # We will multiply on each side to keep symmetry

    P00 = V2*Pxx([[0, 0], [0, 0]])
    P10 = V2*Pxx([[0, 0], [1, 0]])
    P01 = V2*Pxx([[0, 1], [0, 0]])
    P11 = V2*Pxx([[0, 1], [1, 0]])

    if sigma.size == mesh.nC:  # Isotropic!
        sigma = mkvc(sigma)  # ensure it is a vector.
        Sigma = sdiag(np.r_[sigma, sigma])
    elif sigma.shape[1] == 2:  # Diagonal tensor
        Sigma = sdiag(np.r_[sigma[:, 0], sigma[:, 1]])
    elif sigma.shape[1] == 3:  # Fully anisotropic
        row1 = sp.hstack((sdiag(sigma[:, 0]), sdiag(sigma[:, 2])))
        row2 = sp.hstack((sdiag(sigma[:, 2]), sdiag(sigma[:, 1])))
        Sigma = sp.vstack((row1, row2))

    A = P00.T*Sigma*P00 + P10.T*Sigma*P10 + P01.T*Sigma*P01 + P11.T*Sigma*P11
    P = [P00, P10, P01, P11]
    if returnP:
        return A, P
    else:
        return A


if __name__ == '__main__':
    from TensorMesh import TensorMesh
    h = [np.array([1, 2, 3, 4]), np.array([1, 2, 1, 4, 2]), np.array([1, 1, 4, 1])]
    mesh = TensorMesh(h)
    mu = np.ones((mesh.nC, 6))
    A, P = mesh.getFaceInnerProduct(mu, returnP=True)
    B, P = mesh.getEdgeInnerProduct(mu, returnP=True)
