import numpy as np
from BaseMesh import BaseMesh
from utils import ndgrid, mkvc,sdiag,speye,spzeros
from scipy import sparse as sp
from TensorMesh import TensorMesh
import matplotlib.pyplot as plt
import matplotlib
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
    _meshType = 'TetraMesh'

    def __init__(self, h, x0=None):
        super(TetraMesh, self).__init__(np.array([x.size for x in h]), x0)

        assert self.dim>1, "Dimension must be 2 or 3."
        assert len(h) == len(self.x0), "Dimension mismatch. x0 != len(h)"

        for i, h_i in enumerate(h):
            assert type(h_i) == np.ndarray, ("h[%i] is not a numpy array." % i)
            assert len(h_i.shape) == 1, ("h[%i] must be a 1D numpy array." % i)

        # Ensure h contains 1D vectors
        self._h = [mkvc(x) for x in h]

    def __str__(self):
        outStr = '  ---- {0:d}-D TetraMesh ----  '.format(self.dim)
        return outStr
    def nodes():
        doc = "node list as a numpy array "
        def fget(self):
            if self._nodes is None:
                Mesh = TensorMesh(self.h,self.x0)
                if self.dim==2:
                    self._nodes = np.vstack((Mesh.gridN,Mesh.gridCC))
                elif self.dim==3:
                    self._nodes = np.vstack((Mesh.gridN,Mesh.gridFx,Mesh.gridFy,Mesh.gridFz,Mesh.gridCC))
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
        doc = "h is a list containing the cell widths of the underlying tensor mesh in each dimension."
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


    def boundaryIndices():
        doc = "indices of nodes lying on the boundary"
        def fget(self):
            if self._boundaryIndices is None:
                idx = np.arange(np.prod(self.n+1)).reshape(self.n+1,order='F')
                if self.dim==2:
                    idx = np.r_[idx[[0,-1],:].flatten(),idx[:,[0,-1]].flatten()]
                elif self.dim==3:
                    idx = np.r_[idx[[0,-1],:,:].flatten(),idx[:,[0,-1],:].flatten(),idx[:,:,[0,-1]].flatten()]

                self._boundaryIndices = np.unique(idx)
            return self._boundaryIndices
        return locals()
    _boundaryIndices = None  # Store grid by default
    boundaryIndices = property(**boundaryIndices())

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

    def PE1():
        doc = "Projector on first edge of tetrahedra."
        def fget(self):
            if self._PE1 is None:
                if self.dim==2:
                    P = self.P1-self.P3
                else:
                    P = self.P1-self.P4
                self._PE1 = P
            return self._PE1
        return locals()
    _PE1 = None  # Store grid by default
    PE1 = property(**PE1())

    def PE2():
        doc = "Projector on second edge of tetrahedra."
        def fget(self):
            if self._PE2 is None:
                if self.dim==2:
                    P = self.P2-self.P3
                else:
                    P = self.P2-self.P4
                self._PE2 = P
            return self._PE2
        return locals()
    _PE2 = None  # Store grid by default
    PE2 = property(**PE2())

    def PE3():
        doc = "Projector on third edge of tetrahedra."
        def fget(self):
            if self._PE3 is None:
                if self.dim==3:
                    self._PE3 = self.P3-self.P4
            return self._PE3
        return locals()
    _PE3 = None  # Store grid by default
    PE3 = property(**PE3())

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

    def PB():
        doc = "Projector on boundary nodes."
        def fget(self):
            if self._PB is None:
                P = speye(self.nNodes)
                self._PB = P[self.boundaryIndices,:]
            return self._PB
        return locals()
    _PB = None  # Store grid by default
    PB = property(**PB())

    # --------------- Differential Operators --------
    def Dx():
        doc = "Derivative in x-direction"
        def fget(self):
            if self._Dx is None:
                if self.dim==2:
                    e1 =  self.PE1*self.nodes
                    e2 =  self.PE2*self.nodes

                    # compute gradients of basis functions
                    dphi1 =  e2[:,1]/(2*self.vol)
                    dphi2 = -e1[:,1]/(2*self.vol)
                    dphi3 = -dphi1-dphi2


                    self._Dx = sdiag(dphi1)*self.P1 +  sdiag(dphi2)*self.P2  + sdiag(dphi3)*self.P3

                elif self.dim==3:

                    # compute edges
                    e1   =  self.PE1*self.nodes
                    e2   =  self.PE2*self.nodes
                    e3   =  self.PE3*self.nodes
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
                    e1 =  self.PE1*self.nodes
                    e2 =  self.PE2*self.nodes

                    # compute gradients of basis functions
                    dphi1 =  -e2[:,0]/(2*self.vol)
                    dphi2 =   e1[:,0]/(2*self.vol)
                    dphi3 = -dphi1-dphi2


                    self._Dy = sdiag(dphi1)*self.P1 +  sdiag(dphi2)*self.P2  + sdiag(dphi3)*self.P3

                elif self.dim==3:
                     # compute edges
                    e1   =  self.PE1*self.nodes
                    e2   =  self.PE2*self.nodes
                    e3   =  self.PE3*self.nodes
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
                    e1   =  self.PE1*self.nodes
                    e2   =  self.PE2*self.nodes
                    e3   =  self.PE3*self.nodes
                    # compute inverse transformation to reference element
                    cofA = np.c_[e2[:,0]*e3[:,1]-e2[:,1]*e3[:,0], -(e1[:,0]*e3[:,1]-e1[:,1]*e3[:,0]), e1[:,0]*e2[:,1]-e1[:,1]*e2[:,0] ]


                    detA = e1[:,2]*cofA[:,0] + e2[:,2]*cofA[:,1]+ e3[:,2]*cofA[:,2]


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
                    self._GRAD = sp.vstack((self.Dx,self.Dy))
                elif self.dim==3:
                    self._GRAD = sp.vstack((self.Dx,self.Dy,self.Dz))
            return self._GRAD
        return locals()
    _GRAD = None  # Store grid by default
    GRAD = property(**GRAD())

    # --------------- Geometries ---------------------
    def vol():
        doc = "tetrahedra volumes as 1d array."
        def fget(self):
            if(self._vol is None):
                self._vol = self.getVolume(self.nodes,False)
            return self._vol
        return locals()
    _vol = None
    vol = property(**vol())

    # ---------------  Volume and Determinant ---------------------
    def getVolume(self,yc,doDerivative=True):
        assert type(doDerivative) == bool, "doDerivative must be a boolean."
        assert type(yc) == np.ndarray, "yc must be a numpy array."
        assert yc.size == self.nodes.size, "yc is of incorrect size."

        yc = yc.reshape(-1,self.dim,order='F')

        if self.dim == 2:
            PE1 = self.PE1
            PE2 = self.PE2

            e1 = PE1*yc
            e2 = PE2*yc

            V  = (e1[:,0]*e2[:,1] - e1[:,1]*e2[:,0])/2.0

            if doDerivative:
                dV =   sp.hstack(( sdiag(e2[:,1])*PE1 - sdiag(e1[:,1])*PE2 , sdiag(e1[:,1])*PE2 - sdiag(e2[:,0])*PE1))/2.0
                return V,dV
            else:
                return V
        elif self.dim == 3:
            PE1 = self.PE1
            PE2 = self.PE2
            PE3 = self.PE3

            e1 = PE1*yc
            e2 = PE2*yc
            e3 = PE3*yc

            cof11 =    e2[:,1]*e3[:,2] - e2[:,2]*e3[:,1]
            cof21 =  -(e2[:,0]*e3[:,2] - e2[:,2]*e3[:,0])
            cof31 =    e2[:,0]*e3[:,1] - e2[:,1]*e3[:,0]

            V     = (e1[:,0]*cof11 + e1[:,1]*cof21 + e1[:,2]*cof31)/6.0

            if doDerivative:
                Z = spzeros(self.nTetra,self.nNodes)
                # derivatives of cofactors
                dcof11 =   sp.hstack((Z,  sdiag(e3[:,2])*PE2 - sdiag(e2[:,2])*PE3, sdiag(e2[:,1])*PE3 - sdiag(e3[:,1])*PE2))
                dcof21 = - sp.hstack((sdiag(e3[:,2])*PE2 - sdiag(e2[:,2])*PE3,  Z, sdiag(e2[:,0])*PE3 - sdiag(e3[:,0])*PE2))
                dcof31 =   sp.hstack((sdiag(e3[:,1])*PE2 - sdiag(e2[:,1])*PE3, sdiag(e2[:,0])*PE3 - sdiag(e3[:,0])*PE2,Z))

                # apply product rule
                dV  = (1/6.0)*(sp.hstack((sdiag(cof11)*PE1, sdiag(cof21)*self.PE1, sdiag(cof31)*self.PE1) + sdiag(e1[:,0]) * dcof11 + sdiag(e1[:,1]) * dcof21 + sdiag(e1[:,2]) * dcof31))

                return V,dV
            else:
                return V

    def getCofactor(self,yc,doDerivative=True):
        assert type(doDerivative) == bool, "doDerivative must be a boolean."
        assert type(yc) == np.ndarray, "yc must be a numpy array."
        assert yc.size  == self.nodes.size, "yc is of incorrect size."
        assert self.dim == 3, "cofactor exists only for 3D displacements."

        yc = yc.reshape(-1,self.dim,order='F')

        Dx = self.Dx
        Dy = self.Dy
        Dz = self.Dz

        e1 = Dx*yc
        e2 = Dy*yc
        e3 = Dz*yc

        cof11 =    e2[:,1]*e3[:,2] - e2[:,2]*e3[:,1]
        cof21 =  -(e2[:,0]*e3[:,2] - e2[:,2]*e3[:,0])
        cof31 =    e2[:,0]*e3[:,1] - e2[:,1]*e3[:,0]

        cof12 =  -(e1[:,1]*e3[:,2] - e1[:,2]*e3[:,1])
        cof22 =    e1[:,0]*e3[:,2] - e1[:,2]*e3[:,0]
        cof32 =  -(e1[:,0]*e3[:,1] - e1[:,1]*e3[:,0])

        cof13 =    e1[:,1]*e2[:,2] - e1[:,2]*e2[:,1]
        cof23 =  -(e1[:,0]*e2[:,2] - e1[:,2]*e2[:,0])
        cof33 =    e1[:,0]*e2[:,1] - e1[:,1]*e2[:,0]

        cof   = np.c_[cof11,cof21,cof31,cof12,cof22,cof32,cof13,cof23,cof33]

        if doDerivative:
            Z = spzeros(self.nTetra,self.nNodes)
            # derivatives of cofactors
            dcof11 =   sp.hstack((Z,  sdiag(e3[:,2])*Dy - sdiag(e2[:,2])*Dz, sdiag(e2[:,1])*Dz - sdiag(e3[:,1])*Dy))
            dcof21 = - sp.hstack((sdiag(e3[:,2])*Dy - sdiag(e2[:,2])*Dz,  Z, sdiag(e2[:,0])*Dz - sdiag(e3[:,0])*Dy))
            dcof31 =   sp.hstack((sdiag(e3[:,1])*Dy - sdiag(e2[:,1])*Dz, sdiag(e2[:,0])*Dz - sdiag(e3[:,0])*Dy,Z))

            dcof12 = - sp.hstack((Z,  sdiag(e3[:,2])*Dx - sdiag(e1[:,2])*Dz, sdiag(e1[:,1])*Dz - sdiag(e1[:,1])*Dx))
            dcof22 =   sp.hstack((sdiag(e3[:,2])*Dx - sdiag(e1[:,2])*Dz,  Z, sdiag(e1[:,0])*Dz - sdiag(e3[:,0])*Dx))
            dcof32 = - sp.hstack((sdiag(e3[:,1])*Dx - sdiag(e1[:,1])*Dz, sdiag(e1[:,0])*Dz - sdiag(e3[:,0])*Dx,Z))

            dcof13 =   sp.hstack((Z,  sdiag(e2[:,2])*Dx - sdiag(e1[:,2])*Dy, sdiag(e2[:,1])*Dx - sdiag(e2[:,1])*Dx))
            dcof23 = - sp.hstack((sdiag(e2[:,2])*Dx - sdiag(e1[:,2])*Dy,  Z, sdiag(e1[:,0])*Dy - sdiag(e2[:,0])*Dx))
            dcof33 =   sp.hstack((sdiag(e2[:,1])*Dx - sdiag(e1[:,1])*Dy, sdiag(e1[:,0])*Dy - sdiag(e2[:,0])*Dx,Z))
          # apply product rule
            dcof  = [dcof11,dcof21,dcof31,dcof12,dcof22,dcof32,dcof13,dcof23,dcof33]

            return cof,dcof
        else:
            return cof




    def getDeterminant(self, yc, doDerivative=True):
        assert type(doDerivative) == bool, "doDerivative must be a boolean."
        assert type(yc) == np.ndarray, "yc must be a numpy array."
        assert yc.size == self.nodes.size, "yc is of incorrect size."

        yc = yc.reshape(-1,self.dim,order='F')

        if self.dim == 2:
            Dx = self.Dx
            Dy = self.Dy

            e1 = Dx*yc
            e2 = Dy*yc

            V  = (e1[:,0]*e2[:,1] - e1[:,1]*e2[:,0])

            if doDerivative:
                dV =   sp.hstack(( sdiag(e2[:,1])*Dx - sdiag(e1[:,1])*Dy , sdiag(e1[:,1])*Dy - sdiag(e2[:,0])*Dx))
                return V,dV
            else:
                return V
        elif self.dim == 3:
            Dx = self.Dx
            Dy = self.Dy
            Dz = self.Dz

            e1 = Dx*yc
            e2 = Dy*yc
            e3 = Dz*yc

            cof11 =    e2[:,1]*e3[:,2] - e2[:,2]*e3[:,1]
            cof21 =  -(e2[:,0]*e3[:,2] - e2[:,2]*e3[:,0])
            cof31 =    e2[:,0]*e3[:,1] - e2[:,1]*e3[:,0]

            V     = (e1[:,0]*cof11 + e1[:,1]*cof21 + e1[:,2]*cof31)

            if doDerivative:
                Z = spzeros(self.nTetra,self.nNodes)
                # derivatives of cofactors
                dcof11 =   sp.hstack((Z,  sdiag(e3[:,2])*Dy - sdiag(e2[:,2])*Dz, sdiag(e2[:,1])*Dz - sdiag(e3[:,1])*Dy))
                dcof21 = - sp.hstack((sdiag(e3[:,2])*Dy - sdiag(e2[:,2])*Dz,  Z, sdiag(e2[:,0])*Dz - sdiag(e3[:,0])*Dy))
                dcof31 =   sp.hstack((sdiag(e3[:,1])*Dy - sdiag(e2[:,1])*Dz, sdiag(e2[:,0])*Dz - sdiag(e3[:,0])*Dy,Z))

                # apply product rule
                dV  = sp.hstack(( sdiag(cof11)*Dx, sdiag(cof21)*self.Dx, sdiag(cof31)*self.Dx)) + sdiag(e1[:,0]) * dcof11 + sdiag(e1[:,1]) * dcof21 + sdiag(e1[:,2]) * dcof31

                return V,dV
            else:
                return V

    # ---------------  Plotting ---------------------
    def plotMesh(self,yc=None,ax=None):
        """
            plotMesh(self,yc=None,ax=None):

            Plots tetrahedral mesh (currently only 2D supported). Provide shifted nodes when visualizing a deformation.
            If yc is None the Mesh nodes are used.
        """

        if yc is None:
            yc = self.nodes
        else:
            assert type(yc) == np.ndarray, "yc must be a numpy array."
            assert yc.size == self.nodes.size, "yc is of ncorrect size."
            yc = yc.reshape(-1,self.dim,order='F')

        assert self.dim == 2, "dimension must be 2."
        if ax is None:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111)
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        if self.dim == 2:
            ax.set_aspect('equal')
            ax.triplot(yc[:,0], yc[:,1], self.tetra, 'bo-')
            ax.set_xlabel('y')
            ax.set_ylabel('x')

    def plotImage(self,C,yc=None,ax=None):
        """
            plotImage(self,C,yc=None,ax=None):

            Visualizes image C on a possibly deformed tetrahedral mesh given by the nodes yc.
            If yc is None, the nodes of the Mesh are used.

            Colordata can be given either on the nodes or on the tetrahedra.

            see also matplotlib.tripcolor
        """

        assert type(C) == np.ndarray, "C must be a numpy array."
        assert C.size in [self.nNodes, self.nTetra]  , "C must be given either on the nodes or on the tetrahedra."

        if yc is None:
            yc = self.nodes
        else:
            assert type(yc) == np.ndarray, "yc must be a numpy array."
            assert yc.size == self.nodes.size, "yc is of ncorrect size."
            yc = yc.reshape(-1,self.dim)

        assert self.dim == 2, "dimension must be 2."
        if ax is None:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111)
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        if self.dim == 2:
            ax.set_aspect('equal')
            ax.tripcolor(yc[:,0], yc[:,1], self.tetra, C,edgecolors='b')
            ax.set_xlabel('x')
            ax.set_ylabel('y')





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


    M.plotMesh

    print M.GRAD
