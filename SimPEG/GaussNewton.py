import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg.dsolve as dsl
from pylab import norm
from utils import mkvc

def GaussNewton(misfun, regfun, regpar, x0,maxIter=20, maxIterLS=10, LSreduction=1e-4, tolJ=1e-3, tolX=1e-3,
                                                    tolG=1e-3, eps=1e-16, xStop=[]):
    """
        GaussNewton Optimization

        Input:
        ------
            misfun - objective Function (lambda function)
            x0   - starting guess

        Output:
        -------
            xOpt - numerical optimizer
    """
    # initial output
    print "%s GaussNewton %s" % ('='*30,'='*30)
    print "iter\tJc\t\t\tMc\t\t\t\tRc\t\t\t\tnorm(dJ)\tLS"
    print "%s" % '-'*73

    # evaluate stopping criteria
    if xStop==[]:
        xStop=x0

    # initialize
    xc = x0
    STOP = np.zeros((2,1),dtype=bool)
    iterLS = 0; iter = 0
    xOld = xc
    while 1:
        # evaluate objective function
        Mc,dMc,JMc,r = misfun(xc)
        Rc,dRc,HRc = regfun(xc)
        Jc  = Mc + regpar*Rc
        dJc = dMc + regpar*dRc

        print "%3d\t%1.2e\t\t%1.2e\t\t%1.2e\t\t%1.2e\t%d" % (iter, Jc,Mc,regpar*Rc,norm(dJc),iterLS)

        # get search direction
        dx = incg(JMc,HRc,regpar,-dJc,maxIter=10,tol=1e-4,verbose=True)

        # Armijo linesearch
        descent = np.vdot(dJc,dx)
        LS = 0; muLS = 1; iterLS = 1
        while  (iterLS<maxIterLS):
            xt = xc + muLS*dx
            Mt = misfun(xt)
            Rt = regfun(xt)
            Jt  = Mt[0] + regpar*Rt[0]

            LS = Jt < Jc + muLS * LSreduction*descent
            if LS:
                break
            iterLS = iterLS+1
            muLS = .5*muLS
        # ENDWHILE FOR LINE SEARCH

        # store old values
        if not(LS):
            print "line search break"
            return xc
            #raise Exception('Line search fail')
        # update
        xc = xt
        iter = iter + 1

        # Stopping criteria
        STOP[0] = (norm(xc-xOld) < tolX)
        STOP[1] = (maxIter < iter)

        if any(STOP):
            break
        Jold = Jc; xOld = xc
    # ENDWHILE FOR GAUSS NEWTON

    print "%s STOP! %s" % ('-'*33,'-'*33)
    print "%d : |xc-xOld| = %1.4e\t<=\ttolX    \t= %1.4e"  % (STOP[0],norm(xc-xOld),tolX)
    print "%d : iter      = %3d\t\t\t<=\tmaxIter\t\t= %3d" % (STOP[1],iter,maxIter)
    print "%s DONE! %s\n" % ('='*33,'='*33)

    return xc

 # END GAUSS NEWTON

def incg(JMc,HRc,regpar,rhs,maxIter=10,tol=1e-2,verbose=False):
    """
        docstring missing!
    """

    # Prepare for CG iteration.
    x = np.zeros(rhs.shape)
    r = rhs
    rho = np.vdot(r,r)

    iter = 1
    STOP = np.zeros((2,1),dtype=bool)
    while 1:
        # preconditioning step %%%%%%
        z = dsl.spsolve(HRc, r)

        rho1 = rho
        rho = np.vdot(r,z)
        if iter == 1:
            rho0 = norm(z)
            p = z
        if ( iter > 1 ):
            beta  = rho / rho1
            p = z + beta*p
        # endif
        #  Matrix times a vector

        q = JMc(JMc(p,'forward'),'trans') + regpar*HRc.dot(p)   #A(p);

        alpha = rho / np.vdot(p,q)
        x = x + alpha * p                    # update approximation vector

        r = r - alpha*q                      # compute residual
        err = norm( r ) / rho0               # check convergence

        STOP[0] = (err<=tol)
        STOP[1] = maxIter<=iter

        if any(STOP):
            break
        iter = iter+1
        # endif

    #endfor
    if verbose:
        if STOP[0]:
            print 'incg converged in %d iterations. relres=%1.4e\t<=\ttol=%1.4e.' % (iter,err,tol)
        else:
            print 'incg did NOT converge to tol %1.4e. The %dth iterate has relres=%1.4e.' % (tol,iter,err)

    return x



# d = rhs
# normr2 = np.vdot(d,d)
# Iterate.
# for j in xrange(iter):
#     Ad = JMc(d,'forward')
#     alpha = normr2/(np.vdot(Ad,Ad))
#     x  = x + alpha*d
#     r  = r - alpha*Ad
#     s  = JMc(r,'trans')
#     normr2_new = np.vdot(s,s)
#     beta = normr2_new/normr2
#     normr2 = normr2_new
#     d = s + beta*d

def simpleMisfun(x,b):
    """
        simple misfit function

        Mc(x) = .5* || x-b ||^2
    """

    r   = x-b
    Mc  = .5*np.vdot(r,r)
    dMc = r
    def JMc(vec,flag):
        if flag=='forward':
            return vec
        elif flag=='trans':
            return vec

    return Mc,dMc,JMc,r

def simpleRegfun(x,mesh):
    """
        simple regularization function

        Rc(x) = .5* || GRAD*x ||
    """

    GRAD = mesh.cellGrad
    H    = GRAD.T*GRAD
    dRc  = H.dot(x)
    Rc   = .5*np.vdot(x,dRc)

    return Rc,dRc,H


def checkDerivative(fctn,x0):
    """
        Basic derivative check

        Compares error decay of 0th and 1st order Taylor approximation at point
        x0 for a randomized search direction.

       Input:
       ------
         fctn  -  function handle
         x0    -  point at which to check derivative
    """

    print "%s checkDerivative %s" % ('='*20,'='*20)
    print "iter\th\t\t|J0-Jt|\t\t|J0+h*dJ'*dx-Jt|"

    Jc,dJ,H = fctn(x0)

    dx = np.random.randn(len(x0))

    t  = np.logspace(-1,-10,10)
    E0 = np.zeros(t.shape)
    E1 = np.zeros(t.shape)

    for i in range(0,10):
        Jt = fctn(x0+t[i]*dx)

        E0[i] = norm(Jt[0]-Jc)                           # 0th order Taylor
        E1[i] = norm(Jt[0]-Jc-t[i]*np.vdot(dJ.T,dx))      # 1st order Taylor

        print "%d\t%1.2e\t%1.3e\t%1.3e" % (i,t[i],E0[i],E1[i])

    print "%s DONE! %s\n" % ('='*25,'='*25)
    plt.figure()
    plt.clf()
    plt.loglog(t,E0,'b')
    plt.loglog(t,E1,'g--')
    plt.title('checkDerivative')
    plt.xlabel('h')
    plt.ylabel('error of Taylor approximation')
    plt.legend(['0th order', '1st order'],loc='upper left')
    plt.show()
    return

if __name__ == '__main__':
    from TensorMesh import TensorMesh

    h = [25*np.ones(8), 25*np.ones(8), 25*np.ones(8)]
    mesh = TensorMesh(h)

    b = np.sin(mesh.gridCC[:,0]) * np.sin(mesh.gridCC[:,1]) * np.sin(mesh.gridCC[:,2])

    def misfun(x):
        Mc,dMc,JMc,r = simpleMisfun(x,b)
        return Mc,dMc,JMc,r
    def regfun(x):
        Rc,dRc,H = simpleRegfun(x,mesh)
        return Rc,dRc,H


    x0 = np.zeros(mesh.nC)
    xOpt = GaussNewton(misfun,regfun,1e-2,x0,maxIter=100)

