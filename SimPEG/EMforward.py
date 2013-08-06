import numpy as np
from utils import mkvc
from sputils import sdiag
import scipy.sparse.linalg.dsolve as dsl
from InnerProducts import getFaceInnerProduct, getEdgeInnerProduct

def getMisfit(m,mesh,forward,param):

    mu0    = 4*np.pi*1e-7
    omega  = forward['omega'] #[param['indomega']]
    rhs    = forward['rhs']   #[:,param['indrhs']]
    dobs   = param['dobs']   #[:,param['indrhs']]
    Act    = param['Act']
    sigma0 = 1e-8

    mis   = 0
    dmis  = m*0

    # Maxwell's system for E
    for i in range(len(omega)):
        for j in range(rhs.shape[1]):
            Curl  = mesh.edgeCurl
            #Grad  = mesh.nodalGrad
            sigma = Act.T.dot(np.exp(m)) + sigma0
            Me,PP = getEdgeInnerProduct(mesh,sigma)
            Mf,QQ = getFaceInnerProduct(mesh,1/mu0*np.ones(mesh.nC))   # assume mu = mu0
            A = Curl.T.dot(Mf.dot(Curl)) - 1j * omega[i] * Me
            b = -1j*omega[i]*mkvc(np.array(rhs[:,j]),1)
            e = dsl.spsolve(A,b)
            e = mkvc(e)
            print np.linalg.norm(A*e-b)/np.linalg.norm(b)
            P = forward['projection']
            d = P.dot(e)
            
            # change to r[j,:,i] = mkvc(d) - mkvc(dobs[j,:,i])
            # and after computing
            # rw = hadamard(W[j,:,i],r)
            # not to forget to multiply again with W for the gradient 
            r = mkvc(d) - mkvc(dobs[j,:,i])

            mis  = mis + 0.5*np.real(r.conj().T.dot(r))

            # get derivatives
            lam  = dsl.spsolve(A.T,P.T.dot(r))
            lam  = mkvc(lam)
            Gij  = 0
            I3   = sp.vstack((sp.eye(mesh.nC),sp.eye(mesh.nC),sp.eye(mesh.nC)))
            for jj in range(0,8):
                Gij  =  Gij - 1j * omega[i] * (PP[jj].T.dot(sdiag(PP[jj].dot(e)))).dot(I3) 
             
            # keep some of the G's for stochastic gradient
            # if ij == ...
            #   G = vstack((G,Gij))  
            dmis = dmis + np.real(sdiag(np.exp(m)).dot(Act.dot(Gij.conj().T.dot(lam))))


    return mis, dmis, d


    
if __name__ == '__main__':
    from TensorMesh import TensorMesh 
    from interpmat import interpmat

    from scipy import sparse as sp

    # generate the mesh
    h = [25*np.ones(16),25*np.ones(16),25*np.ones(16)]
    mesh = TensorMesh(h,[-200.0,-200.0,-200.0])

    # generate the interpolation matrix
    xs = np.array([-40.1,-43.2,54.6,65.8])
    ys = np.array([-32.1,-41.3,54.2,62.1])
    zs = np.array([0.0,0.0,0.0,0.0]);

    xyz = mesh.gridEx
    x   = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    x   = list(set(x)); y = list(set(y)); z = list(set(z))
    Px  = interpmat(x,y,z,xs,ys,zs)
    xyz = mesh.gridEy
    x   = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    x   = list(set(x)); y = list(set(y)); z = list(set(z))
    Py  = interpmat(x,y,z,xs,ys,zs)
    xyz = mesh.gridEz
    x   = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    x   = list(set(x)); y = list(set(y)); z = list(set(z))
    Pz  = interpmat(x,y,z,xs,ys,zs)
    P   = sp.block_diag((Px,Py,0*Pz))

    # generate sources (waiting to integrate this with Chris's sources)
    numsrc  = 3
    ne      = np.sum(mesh.nE)
    Q       = np.matrix(np.random.randn(ne,numsrc))

    omega   = [10,100]
    forward = {'omega':omega, 'rhs':Q,'projection':P}
    dobs    = np.ones([numsrc,3*np.size(xs),np.size(omega)])
    

    # generate matrix for active cells
    xyz    = mesh.gridCC
    x      = xyz[:,0]; y = xyz[:,1]; z = xyz[:,2]
    a      = z*0
    a[z<0] = 1
    nact   = sum(a)
    I      = np.nonzero(a>0)[0]
    Act    = sp.coo_matrix((I*0+1.0,(I,I)),shape=(nact,mesh.nC))

    param   = {'dobs':dobs,'Act':Act}

    # setup the model
    m = np.log(1e-3*np.ones(nact))
    # solve maxwell and get derivatives
    mis, dmis, d = getMisfit(m,mesh,forward,param)

    # check derivatives
    dm = 1e-1*np.random.randn(nact)
    mis1, dmis1, d1 = getMisfit(m+dm,mesh,forward,param)

    print mis1-mis,  mis1-mis - dm.dot(dmis)


