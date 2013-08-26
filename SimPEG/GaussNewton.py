import numpy as np
import matplotlib.pyplot as plt
from pylab import norm

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
    print "%s GaussNewton %s" % ('='*22,'='*22)
    print "iter\tJc\tMc\t\tRc\tnorm(dJ)\tLS" 
    print "%s" % '-'*57
    
    # evaluate stopping criteria
    if xStop==[]:
        xStop=x0
    
    # initialize
    xc = x0
    STOP = np.zeros((5,1),dtype=bool)
    iterLS = 0; iter = 0
    
    xOld = xc
    while 1:
        # evaluate objective function
        Mc,dMc,JMc,d = misfun(xc) 
        Rc,dRc,HRc = regfun(xc)
        Jc  = Mc + regpar*Rc
        dJc = dMc + regpar*dRc

        print "%3d\t%1.2e\t%1.2e\t%d" % (iter, Jc[0],norm(dJc),iterLS)
                
        # get search direction
        r  = dobs - d
        dx = incg(JMc,HRc,regpar,-dJc,r,iter=10,tolLin=1e-2)
        
        # Armijo linesearch
        descent = np.vdot(dJc,dx)
        LS = 0; muLS = 1; iterLS = 1
        while  (iterLS<maxIterLS):
            xt = xc + muLS*dx
            Mt = misfun(xt) 
            Rt = regfun(xt)
            Jt  = Mt + regpar*Rt

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
        Jold = Jc; xOld = xc
        # update 
        xc = xt
        iter = iter + 1

        # Stopping criteria
        STOP[0] = (norm(xc-xt) < tolX)
        STOP[1] = (maxIter < iter)

        if any(STOP):
            break
        # ENDWHILE FOR GAUSS NEWTON
        
    print "%s STOP! %s" % ('-'*25,'-'*25)
    print "%d : |xc-xOld| = %1.4e <= tolX    = %1.4e"  % (STOP[0],norm(xc-xOld),tolX)
    print "%d : iter      = %3d\t <= maxIter\t       = %3d"     % (STOP[1],iter,maxIter)
    print "%s DONE! %s\n" % ('='*25,'='*25)
    
    return xc
      
 # END GAUSS NEWTON
 
def incg(JMc,HRc,regpar,rhs,maxIter=10,tolLin=1e-2):     

# Prepare for CG iteration.
x = zeros(shape(rhs))
r = rhs
n = length(b)
rho = np.vdot(r,r)
  
for iter in xrange(maxIter):

     # preconditioning step %%%%%%
     z = sp.dsolve(HRc,r)
     
     rho1 = rho
     rho = np.vdot(r,z)
     if iter == 1: 
        rho0 = norm(z)
     #end   

     if ( iter > 1 ):
        beta = rho / rho1
        p = z + beta*p
     else:
        p = z
     # endif
     #  Matrix times a vector 
      
     q = JMc(JMc(p,'forward'),'trans') + regpar*HRc.dot(p)   #A(p);
 
     alpha = rho / np.vdot(p,q)
     x = x + alpha * p                    # update approximation vector

     r = r - alpha*q                      # compute residual
     err = norm( r ) / rho0               # check convergence
     numiter = iter;
     
     if (err <= tol):
        break 
     # endif 
     
  #endfor
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
  

def Rosenbrock(x):
    """
        Rosenbrock function for testing GaussNewton scheme
    """
    J   = 100*(x[1]-x[0]**2)**2+(1-x[0])**2 
    dJ  = np.array([2*(200*x[0]**3-200*x[0]*x[1]+x[0]-1),200*(x[1]-x[0]**2)])
    H = np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]],[ -400*x[0], 200]],dtype=float);
    
    return J,dJ,H
    
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
    
    dx = np.random.randn(len(x0),1)
    
    t  = np.logspace(-1,-10,10)
    E0 = np.zeros(t.shape)
    E1 = np.zeros(t.shape)
    
    for i in range(0,10):
        Jt = fctn(x0+t[i]*dx)
        E0[i] = norm(Jt[0]-Jc[0])                           # 0th order Taylor 
        E1[i] = norm(Jt[0]-Jc[0]-t[i]*np.dot(dJ.T,dx))      # 1st order Taylor 
        
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
    x0 = np.array([[2.6],[3.7]])
    fctn = lambda x:Rosenbrock(x)
    checkDerivative(fctn,x0)
    xOpt = GaussNewton(fctn,x0,maxIter=20)
    print "xOpt=[%f,%f]" % (xOpt[0],xOpt[1])
   