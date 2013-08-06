import numpy as np
from scipy import sparse as sp
import scipy.sparse.linalg as spla
#from utils import mkvc
#from sputils import sdiag
#import scipy.sparse.linalg.dsolve as dsl


def solveLinearSystem(M,K,omega,b,G):
    # solve the maxwell system 
    # (K + 1j*omega*M)*u = b
    # G is the null space of K, K*G = 0

    A  = K + 1j*omega*M
    P1 = np.hstack((A + G.dot(G.T), 1j*omega*M.dot(G)))
    P2 = np.hstack((G.T.dot(M),   G.T.dot(M.dot(G))))
    AP = np.vstack((P1,P2))
    R  = sp.vstack( (sp.eye(np.size(b)), 1/1j/omega * G.T))
    RT = sp.hstack((sp.eye(np.size(b)), G))

    M1 = sp.tril(AP)
    M2 = sp.triu(AP)
    D  = A - sp.triu(A,1) - sp.tril(A,-1)
    t  = R.dot(b)
    t  = spla.spsolve(M1, t)
    print np.shape(t)

    #t = RT.dot(spla.spsolve(M2.dot(D.dot(spla.spsolve(M1, R.dot(b))))))
    #def mv(x):
        #    return RT.dot(spla.spsolve(M2.dot(D.dot(spla.spsolve(M1, R.dot(x))))))
            
    #n  = np.shape(A)[0]
    #MV = spla.LinearOperator((n, n), mv)
    #t  = MV*b
    

    #x = spla.bicgstab(A, b, x0=None, tol=1e-9, maxiter=500, xtype=None, M=MV)

    return t

if __name__ == '__main__':
    from TensorMesh import TensorMesh
 
    #from scipy import sparse as sp

    # generate the mesh
    h = [12.5*np.ones(32),12.5*np.ones(32),12.5*np.ones(32)]
    mesh = TensorMesh(h,[-200.0,-200.0,-200.0])
    C = mesh.edgeCurl
    G = mesh.nodalGrad
    M = sp.eye(sum(mesh.nE))
    K = C.T.dot(C);
    
    print np.shape(G.T.dot(M))
    P2 = np.hstack((G.T.dot(M),   G.T.dot(M.dot(G))))

    b = np.random.rand(sum(mesh.nE))
    omega = 10

    x = solveLinearSystem(M,K,omega,b,G)