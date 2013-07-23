import numpy as np
from BaseMesh import BaseMesh
from utils import ndgrid, mkvc
from sputils import sdiag,speye,appendBottom,appendBottom3
class TetraMesh(BaseMesh):
    """
    TetraMesh is a mesh class that deals with structured triangular/tetrahedral elements
    generated on a tensor mesh.

    Any cell that has a constant width along the entire axis
    such that it can defined by a single width vector, called 'h'.

    e.g.

        hx = np.array([1,1,1])
        hy = np.array([1,2])
        hz = np.array([1,1,1,1])

        mesh = TetraMesh([hx, hy, hz])

    """
    def __init__(self, h, x0=None):
        super(TetraMesh, self).__init__(np.array([x.size for x in h]), x0)

        assert self.dim>1, "Dimension must be 2 or 3."
        assert len(h) == len(self.x0), "Dimension mismatch. x0 != len(h)"

        for i, h_i in enumerate(h):
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)

        # Ensure h contains 1D vectors
        self._h = [mkvc(x) for x in h]



    def nodes():
        doc = "node list as a numpy array "
        def fget(self):
            if self._nodes is None:
                if self.dim==2:
                    self._nodes = np.vstack((self.gridN,self.gridCC))
                elif self.dim==3:
                    self._nodes = np.vstack((self.gridN,self.gridFx,self.gridFy,self.gridFz,self.gridCC))
            return self._nodes
        return locals()
    _nodes = None
    nodes = property(**nodes())


    def tetra():
        doc = "tetrahedra list (integer array)"
        def fget(self):
            if self.dim==2:
                idn = np.arange(self.nN).reshape(self.n+1,order='F')
                idn = idn[:-1,:-1].reshape(1,-1,order='F')
                idc = np.arange(self.nC)+self.nN
                iy  = self.n[0]+1
                # define four triangles per pixel
                tet = np.vstack((idn,idn+1,idc))
                tet = np.vstack((tet,idn+1,idn+iy+1,idc))
                tet = np.vstack((tet,idn+iy+1,idn+iy,idc))
                tet = np.vstack((tet,idn+iy,idn,idc))
                self._tetra = tet.reshape(3,-1,order='F').T
            elif self.dim==3:
                idn = np.arange(self.nN).reshape(self.n+1,order='F')
                idn = idn[:-1,:-1,:-1].reshape(1,-1,order='F')
                iFx = (np.arange(np.prod(self.nFx))+self.nN).reshape(self.n+[1,0,0],order='F')
                iFx = iFx[:-1,:,:].flatten(order='F')
                iFy = (np.arange(np.prod(self.nFy))+self.nN+np.prod(self.nFx)).reshape(self.n+[0,1,0],order='F')
                iFy = iFy[:,:-1,:].flatten(order='F')
                iFz = (np.arange(np.prod(self.nFz))+self.nN+np.prod(self.nFx)+np.prod(self.nFy)).reshape(self.n+[0,0,1],order='F')
                iFz = iFz[:,:,:-1].flatten(order='F')
                idc = np.arange(self.nC)+self.nN+np.prod(self.nFx)+np.prod(self.nFy)+np.prod(self.nFz)
                iyn = self.n[0]+1
                izn  = np.prod(self.n[0:2]+1)
                iyc = self.n[0]
                izc  = np.prod(self.n[0:2])

                # define 24 tetrahedra
                tet = np.vstack((    idn          ,idn+1         ,iFy,idc))
                tet = np.vstack((tet,idn+1        ,idn+izn+1     ,iFy,idc))
                tet = np.vstack((tet,idn+izn+1    ,idn+izn       ,iFy,idc))
                tet = np.vstack((tet,idn+izn      ,idn           ,iFy,idc)) # 4
                tet = np.vstack((tet,idn+1        ,idn+iyn+1     ,iFx+1,idc))
                tet = np.vstack((tet,idn+iyn+1    ,idn+iyn+1+izn ,iFx+1,idc))
                tet = np.vstack((tet,idn+iyn+1+izn,idn+1+izn     ,iFx+1,idc))
                tet = np.vstack((tet,idn+1+izn    , idn+1        ,iFx+1,idc)) # 8
                tet = np.vstack((tet,idn+1        ,idn          ,iFz  ,idc))
                tet = np.vstack((tet,idn          ,idn+iyn      ,iFz  ,idc))
                tet = np.vstack((tet,idn+iyn      ,idn+1+iyn    ,iFz  ,idc))
                tet = np.vstack((tet,idn+1+iyn    ,idn+1        ,iFz  ,idc)) #12
                tet = np.vstack((tet,idn+iyn      ,idn          ,iFx  ,idc))
                tet = np.vstack((tet,idn          ,idn+izn      ,iFx  ,idc))
                tet = np.vstack((tet,idn+izn      ,idn+iyn+izn  ,iFx  ,idc))
                tet = np.vstack((tet,idn+iyn+izn  ,idn+iyn      ,iFx  ,idc)) #16
                tet = np.vstack((tet,idn+iyn+1    ,idn+iyn      ,iFy+iyc  ,idc))
                tet = np.vstack((tet,idn+iyn      ,idn+iyn+izn  ,iFy+iyc  ,idc))
                tet = np.vstack((tet,idn+iyn+izn  ,idn+1+iyn+izn,iFy+iyc  ,idc))
                tet = np.vstack((tet,idn+1+iyn+izn,idn+1+iyn     ,iFy+iyc  ,idc)) #20
                tet = np.vstack((tet,idn+izn      ,idn+izn+1     ,iFz+izc  ,idc))
                tet = np.vstack((tet,idn+izn+1    ,idn+izn+1+iyn ,iFz+izc  ,idc))
                tet = np.vstack((tet,idn+izn+1+iyn,idn+izn+iyn   ,iFz+izc  ,idc))
                tet = np.vstack((tet,idn+izn+iyn  ,idn+izn       ,iFz+izc  ,idc))

                self._tetra = tet.reshape(4,-1,order='F').T
            return self._tetra
        return locals()
    _tetra = None
    tetra = property(**tetra())

    def nNodes():
        doc = "Number of nodes (int)"
        def fget(self):
            if self.dim==2:
                return self.nN+self.nC
            elif self.dim==3:
                return self.nN+np.prod(self.nFx)+np.prod(self.nFy)+np.prod(self.nFz)+self.nC
        return locals()
    nNodes = property(**nNodes())

    def nTetra():
        doc = "Number of tetrahedra (int)"
        def fget(self):
            if self.dim==2:
                return 4*self.nC
            elif self.dim==3:
                return 24*self.nC
        return locals()
    nTetra = property(**nTetra())




    def h():
        doc = "h is a list containing the cell widths of the tensor mesh in each dimension."
        fget = lambda self: self._h
        return locals()
    h = property(**h())

    def hx():
        doc = "Width of cells in the x direction"
        fget = lambda self: self._h[0]
        return locals()
    hx = property(**hx())

    def hy():
        doc = "Width of cells in the y direction"
        fget = lambda self: None if self.dim < 2 else self._h[1]
        return locals()
    hy = property(**hy())

    def hz():
        doc = "Width of cells in the z direction"
        fget = lambda self: None if self.dim < 3 else self._h[2]
        return locals()
    hz = property(**hz())

    def vectorNx():
        doc = "Nodal grid vector (1D) in the x direction."
        fget = lambda self: np.r_[0., self.hx.cumsum()] + self.x0[0]
        return locals()
    vectorNx = property(**vectorNx())

    def vectorNy():
        doc = "Nodal grid vector (1D) in the y direction."
        fget = lambda self: None if self.dim < 2 else np.r_[0., self.hy.cumsum()] + self.x0[1]
        return locals()
    vectorNy = property(**vectorNy())

    def vectorNz():
        doc = "Nodal grid vector (1D) in the z direction."
        fget = lambda self: None if self.dim < 3 else np.r_[0., self.hz.cumsum()] + self.x0[2]
        return locals()
    vectorNz = property(**vectorNz())

    def vectorCCx():
        doc = "Cell-centered grid vector (1D) in the x direction."
        fget = lambda self: np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5 + self.x0[0]
        return locals()
    vectorCCx = property(**vectorCCx())

    def vectorCCy():
        doc = "Cell-centered grid vector (1D) in the y direction."
        fget = lambda self: None if self.dim < 2 else np.r_[0, self.hy[:-1].cumsum()] + self.hy*0.5 + self.x0[1]
        return locals()
    vectorCCy = property(**vectorCCy())

    def vectorCCz():
        doc = "Cell-centered grid vector (1D) in the z direction."
        fget = lambda self: None if self.dim < 3 else np.r_[0, self.hz[:-1].cumsum()] + self.hz*0.5 + self.x0[2]
        return locals()
    vectorCCz = property(**vectorCCz())

    def gridCC():
        doc = "Cell-centered grid."

        def fget(self):
            if self._gridCC is None:
                self._gridCC = ndgrid([x for x in [self.vectorCCx, self.vectorCCy, self.vectorCCz] if not x is None])
            return self._gridCC
        return locals()
    _gridCC = None  # Store grid by default
    gridCC = property(**gridCC())

    def gridN():
        doc = "Nodal grid."

        def fget(self):
            if self._gridN is None:
                self._gridN = ndgrid([x for x in [self.vectorNx, self.vectorNy, self.vectorNz] if not x is None])
            return self._gridN
        return locals()
    _gridN = None  # Store grid by default
    gridN = property(**gridN())


    def gridFx():
        doc = "Face staggered grid in the x direction."

        def fget(self):
            if self._gridFx is None:
                self._gridFx = ndgrid([x for x in [self.vectorNx, self.vectorCCy, self.vectorCCz] if not x is None])
            return self._gridFx
        return locals()
    _gridFx = None  # Store grid by default
    gridFx = property(**gridFx())

    def gridFy():
        doc = "Face staggered grid in the y direction."

        def fget(self):
            if self._gridFy is None:
                self._gridFy = ndgrid([x for x in [self.vectorCCx, self.vectorNy, self.vectorCCz] if not x is None])
            return self._gridFy
        return locals()
    _gridFy = None  # Store grid by default
    gridFy = property(**gridFy())

    def gridFz():
        doc = "Face staggered grid in the z direction."

        def fget(self):
            if self._gridFz is None:
                self._gridFz = ndgrid([x for x in [self.vectorCCx, self.vectorCCy, self.vectorNz] if not x is None])
            return self._gridFz
        return locals()
    _gridFz = None  # Store grid by default
    gridFz = property(**gridFz())
    def getBoundaryIndex(self, gridType):
        """Needed for faces edges and cells"""
        pass

    def getCellNumbering(self):
        pass

    # ---------------  Projectors ---------------------
    def P1():
        doc = "Projector on first node of tetrahedra."

        def fget(self):
            if self._P1 is None:
                P = speye(self.nNodes)
                P = P[self.tetra[:,0],:]
                self._P1 = P
            return self._P1
        return locals()
    _P1 = None  # Store grid by default
    P1 = property(**P1())

    def P2():
        doc = "Projector on second node of tetrahedra."

        def fget(self):
            if self._P2 is None:
                P = speye(self.nNodes)
                P = P[self.tetra[:,1],:]
                self._P2 = P
            return self._P2
        return locals()
    _P2 = None  # Store grid by default
    P2 = property(**P2())

    def P3():
        doc = "Projector on third node of tetrahedra."

        def fget(self):
            if self._P3 is None:
                P = speye(self.nNodes)
                P = P[self.tetra[:,2],:]
                self._P3 = P
            return self._P3
        return locals()
    _P3 = None  # Store grid by default
    P3 = property(**P3())

    def P4():
        doc = "Projector on fourth node of tetrahedra."
        def fget(self):
            if self._P4 is None:
                if self.dim==3:
                    P = speye(self.nNodes)
                    P = P[self.tetra[:,3],:]
                    self._P4 = P
            return self._P4
        return locals()
    _P4 = None  # Store grid by default
    P4 = property(**P4())

    def PC():
        doc = "Projector on barycenter of tetrahedra."
        def fget(self):
            if self._PC is None:
                if self.dim==2:
                    self._PC = (1./3)*(self.P1+self.P2+self.P3)
                elif self.dim==3:
                    self._PC = (0.25)*(self.P1+self.P2+self.P3+self.P4)
            return self._PC
        return locals()
    _PC = None  # Store grid by default
    PC = property(**PC())

    # --------------- Differential Operators --------
    def Dx():
        doc = "Derivative in x-direction"
        def fget(self):
            if self._Dx is None:
                if self.dim==2:
                    e1 =  self.P1*self.nodes - self.P3*self.nodes
                    e2 =  self.P2*self.nodes - self.P3*self.nodes

                    # compute gradients of basis functions
                    dphi1 =  e2[:,1]/(2*self.vol)
                    dphi2 = -e1[:,1]/(2*self.vol)
                    dphi3 = -dphi1-dphi2


                    self._Dx = sdiag(dphi1)*self.P1 +  sdiag(dphi2)*self.P2  + sdiag(dphi3)*self.P3

                elif self.dim==3:

                    # compute edges
                    e1   =  self.P1*self.nodes - self.P4*self.nodes
                    e2   =  self.P2*self.nodes - self.P4*self.nodes
                    e3   =  self.P3*self.nodes - self.P4*self.nodes
                    # compute inverse transformation to reference element
                    cofA =  np.c_[ e2[:,1]*e3[:,2]-e2[:,2]*e3[:,1],-(e1[:,1]*e3[:,2]-e1[:,2]*e3[:,1]),e1[:,1]*e2[:,2]-e1[:,2]*e2[:,1]]

                    detA = e1[:,0]*cofA[:,0] + e2[:,0]*cofA[:,1]+ e3[:,0]*cofA[:,2]


                    # compute gradients of basis functions
                    dphi1 =   cofA[:,0]/detA
                    dphi2 =   cofA[:,1]/detA
                    dphi3 =   cofA[:,2]/detA
                    dphi4 =   - dphi1 - dphi2 - dphi3

                    self._Dx =  sdiag(dphi1)*self.P1 + sdiag(dphi2)*self.P2 + sdiag(dphi3)*self.P3 + sdiag(dphi4)*self.P4
            return self._Dx
        return locals()
    _Dx = None  # Store grid by default
    Dx = property(**Dx())

    def Dy():
        doc = "Derivative in y-direction"
        def fget(self):
            if self._Dy is None:
                if self.dim==2:
                    e1 =  self.P1*self.nodes - self.P3*self.nodes
                    e2 =  self.P2*self.nodes - self.P3*self.nodes

                    # compute gradients of basis functions
                    dphi1 =  -e2[:,0]/(2*self.vol)
                    dphi2 =   e1[:,0]/(2*self.vol)
                    dphi3 = -dphi1-dphi2


                    self._Dy = sdiag(dphi1)*self.P1 +  sdiag(dphi2)*self.P2  + sdiag(dphi3)*self.P3

                elif self.dim==3:
                     # compute edges
                    e1   =  self.P1*self.nodes - self.P4*self.nodes
                    e2   =  self.P2*self.nodes - self.P4*self.nodes
                    e3   =  self.P3*self.nodes - self.P4*self.nodes
                    # compute inverse transformation to reference element
                    cofA = np.c_[-(e2[:,0]*e3[:,2]-e2[:,2]*e3[:,0]),e1[:,0]*e3[:,2]-e1[:,2]*e3[:,0],-(e1[:,0]*e2[:,2]-e1[:,2]*e2[:,0])]


                    detA = e1[:,1]*cofA[:,0] + e2[:,1]*cofA[:,1]+ e3[:,1]*cofA[:,2]


                    # compute gradients of basis functions
                    dphi1 =   cofA[:,0]/detA
                    dphi2 =   cofA[:,1]/detA
                    dphi3 =   cofA[:,2]/detA
                    dphi4 =   - dphi1 - dphi2 - dphi3

                    self._Dy =  sdiag(dphi1)*self.P1 + sdiag(dphi2)*self.P2 + sdiag(dphi3)*self.P3 + sdiag(dphi4)*self.P4
            return self._Dy
        return locals()
    _Dy = None  # Store grid by default
    Dy = property(**Dy())

    def Dz():
        doc = "Derivative in z-direction"
        def fget(self):
            if self._Dz is None:
                if self.dim==3:
                     # compute edges
                    e1   =  self.P1*self.nodes - self.P4*self.nodes
                    e2   =  self.P2*self.nodes - self.P4*self.nodes
                    e3   =  self.P3*self.nodes - self.P4*self.nodes
                    # compute inverse transformation to reference element
                    cofA = np.c_[e2[:,0]*e3[:,1]-e2[:,1]*e3[:,0], -(e1[:,0]*e3[:,1]-e1[:,1]*e3[:,0]), e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0] ]


                    detA = e1[:,2]*cofA[:,0] + e2[:,2]*cofA[:,2]+ e3[:,2]*cofA[:,2]


                    # compute gradients of basis functions
                    dphi1 =   cofA[:,0]/detA
                    dphi2 =   cofA[:,1]/detA
                    dphi3 =   cofA[:,2]/detA
                    dphi4 =   - dphi1 - dphi2 - dphi3

                    self._Dz =  sdiag(dphi1)*self.P1 + sdiag(dphi2)*self.P2 + sdiag(dphi3)*self.P3 + sdiag(dphi4)*self.P4
            return self._Dz
        return locals()
    _Dz = None  # Store grid by default
    Dz = property(**Dz())

    def GRAD():
        doc = "Gradient"
        def fget(self):
            if self._GRAD is None:
                if self.dim==2:
                    self._GRAD = appendBottom(self.Dx,self.Dy)
                elif self.dim==3:
                    self._GRAD = appendBottom3(self.Dx,self.Dy,self.Dz)

            return self._GRAD
        return locals()
    _GRAD = None  # Store grid by default
    GRAD = property(**GRAD())

    # --------------- Geometries ---------------------
    def vol():
        doc = "tetrahedra volumes as 1d array."

        def fget(self):
            if(self._vol is None):
                if(self.dim == 2):
                    x1 = self.P1*self.nodes
                    x2 = self.P2*self.nodes
                    x3 = self.P3*self.nodes

                    self._vol = ((x1[:,0]-x3[:,0])*(x2[:,1]-x3[:,1])-(x2[:,0]-x3[:,0])*(x1[:,1]-x3[:,1]))/2.;
                elif(self.dim == 3):
                    x1 = self.P1*self.nodes
                    x2 = self.P2*self.nodes
                    x3 = self.P3*self.nodes
                    x4 = self.P4*self.nodes

                    e1 = x1-x4
                    e2 = x2-x4
                    e3 = x3-x4

                    self._vol = (1./6)* (e1[:,0]*e2[:,1]*e3[:,2] + e2[:,0]*e3[:,1]*e1[:,2] + e3[:,0]*e1[:,1]*e2[:,2] - e1[:,2]*e2[:,1]*e3[:,0] - e2[:,2]*e3[:,1]*e1[:,0] - e3[:,2]*e1[:,1]*e2[:,0])

            return self._vol
        return locals()
    _vol = None
    vol = property(**vol())




    def area():
        pass

    def edge():
        pass


if __name__ == '__main__':
    print('Welcome to TetraMesh!')

    testDim = 3
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

    print M.GRAD
