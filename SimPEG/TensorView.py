import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D


class TensorView(object):
    """
    Provides viewing functions for TensorMesh

    This class is inherited by TensorMesh
    """
    def __init__(self):
        pass

    def plotImage(self, I, imageType='CC', figNum=1,ax=None,direction='z',numbering=True):
        """
        Mesh.plotImage(I)

        Plots scalar fields on the given mesh.

        Input:

            I           - scalar field (np.array)

        Optional Input:

            imageType   - type of image ('CC','N','Fx','Fy','Fz','Ex','Ey','Ez')
            figNum      - number of figure to plot to
            ax          - axis to plot to
            direction   - 3D only. slice dimensions
            numbering   - 3D only. show numbering of slices
        """
        assert type(I) == np.ndarray, "I must be a numpy array"
        assert type(numbering) == bool, "numbering must be a bool"
        assert imageType in ["CC", "N","Fx","Fy","Fz","Ex","Ey","Ez"], "imageType must be 'CC', 'N','Fx','Fy','Fz','Ex','Ey','Ez'"
        assert direction in ["x", "y","z"], "direction must be either x,y, or z"


        if imageType == 'CC':
            assert I.size == self.nC, "Incorrect dimensions for CC."
        elif imageType == 'N':
            assert I.size == self.nN, "Incorrect dimensions for N."
        elif imageType == 'Fx':
            assert I.size == np.prod(self.nFx), "Incorrect dimensions for Fx."
        elif imageType == 'Fy':
            assert I.size == np.prod(self.nFy), "Incorrect dimensions for Fy."
        elif imageType == 'Fz':
            assert I.size == np.prod(self.nFz), "Incorrect dimensions for Fz."
        elif imageType == 'Ex':
            assert I.size == np.prod(self.nEx), "Incorrect dimensions for Ex."
        elif imageType == 'Ey':
            assert I.size == np.prod(self.nEy), "Incorrect dimensions for Ey."
        elif imageType == 'Ez':
            assert I.size == np.prod(self.nEz), "Incorrect dimensions for Ez."

        if ax is None:
            fig = plt.figure(figNum)
            fig.clf()
            ax = plt.subplot(111)
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        if self.dim == 1:
            if imageType == 'CC':
                ph = ax.plot(self.vectorCCx, I, '-ro')
            elif imageType == 'N':
                ph = ax.plot(self.vectorNx, I, '-bs')
            ax.set_xlabel("x")
            ax.axis('tight')
        elif self.dim == 2:
            if imageType == 'CC':
                C = I[:].reshape(self.n, order='F')
            elif imageType == 'N':
                C = I[:].reshape(self.n+1, order='F')
                C = 0.25*(C[:-1, :-1] + C[1:, :-1] + C[:-1, 1:] + C[1:, 1:])
            elif imageType == 'Fx':
                C = I[:].reshape(self.nFx, order='F')
                C = 0.5*(C[:-1, :] + C[1:, :] )
            elif imageType == 'Fy':
                C = I[:].reshape(self.nFy, order='F')
                C = 0.5*(C[:, :-1] + C[:, 1:] )
            elif imageType == 'Ex':
                C = I[:].reshape(self.nEx, order='F')
                C = 0.5*(C[:,:-1] + C[:,1:] )
            elif imageType == 'Ey':
                C = I[:].reshape(self.nEy, order='F')
                C = 0.5*(C[:-1,:] + C[1:,:] )

            ph = ax.pcolormesh(self.vectorNx, self.vectorNy, C.T)
            ax.axis('tight')
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        elif self.dim == 3:
            if direction == 'z':

                # get copy of image and average to cell-centres is necessary
                if imageType == 'CC':
                    Ic = I[:].reshape(self.n, order='F')
                elif imageType == 'N':
                    Ic = I[:].reshape(self.n+1, order='F')
                    Ic = .125*(Ic[:-1,:-1,:-1]+Ic[1:,:-1,:-1] + Ic[:-1,1:,:-1]+ Ic[1:,1:,:-1]+ Ic[:-1,:-1,1:]+Ic[1:,:-1,1:] + Ic[:-1,1:,1:]+ Ic[1:,1:,1:] )
                elif imageType == 'Fx':
                    Ic = I[:].reshape(self.nFx, order='F')
                    Ic = .5*(Ic[:-1,:,:]+Ic[1:,:,:])
                elif imageType == 'Fy':
                    Ic = I[:].reshape(self.nFy, order='F')
                    Ic = .5*(Ic[:,:-1,:]+Ic[:,1:,:])
                elif imageType == 'Fz':
                    Ic = I[:].reshape(self.nFz, order='F')
                    Ic = .5*(Ic[:,:,:-1]+Ic[:,:,1:])
                elif imageType == 'Ex':
                    Ic = I[:].reshape(self.nEx, order='F')
                    Ic = .25*(Ic[:,:-1,:-1]+Ic[:,1:,:-1]+Ic[:,:-1,1:]+Ic[:,1:,:1])
                elif imageType == 'Ey':
                    Ic = I[:].reshape(self.nEy, order='F')
                    Ic = .25*(Ic[:-1,:,:-1]+Ic[1:,:,:-1]+Ic[:-1,:,1:]+Ic[1:,:,:1])
                elif imageType == 'Ez':
                    Ic = I[:].reshape(self.nEz, order='F')
                    Ic = .25*(Ic[:-1,:-1,:]+Ic[1:,:-1,:]+Ic[:-1,1:,:]+Ic[1:,:1,:])

                # determine number oE slices in x and y dimension
                nX = np.ceil(np.sqrt(self.nCz))
                nY = np.ceil(self.nCz/nX)

                #  allocate space for montage
                C = np.zeros((nX*self.nCx,nY*self.nCz))


                nCx = self.nCx
                nCy = self.nCy
                for iy in range(int(nY)):
                    for ix in range(int(nX)):
                        iz = ix + iy*nX
                        if iz < self.nCz:
                            C[ix*nCx:(ix+1)*nCx, iy*nCy:(iy+1)*nCy] = Ic[:, :, iz]
                        else:
                            C[ix*nCx:(ix+1)*nCx, iy*nCy:(iy+1)*nCy] = np.nan

                C = np.ma.masked_where(np.isnan(C), C)
                xx = np.r_[0, np.cumsum(np.kron(np.ones((nX, 1)), self.hx).ravel())]
                yy = np.r_[0, np.cumsum(np.kron(np.ones((nY, 1)), self.hy).ravel())]
                ph = ax.pcolormesh(xx, yy, C.T)
                # Plot the lines
                gx =  np.arange(nX+1)*self.vectorNx[-1]
                gy =  np.arange(nY+1)*self.vectorNy[-1]
                # Repeat and seperate with NaN
                gxX = np.c_[gx, gx, gx+np.nan].ravel()
                gxY = np.kron(np.ones((nX+1, 1)), np.array([0, sum(self.hy)*nY, np.nan])).ravel()
                gyX = np.kron(np.ones((nY+1, 1)), np.array([0, sum(self.hx)*nX, np.nan])).ravel()
                gyY = np.c_[gy, gy, gy+np.nan].ravel()
                ax.plot(gxX, gxY, 'w-', linewidth=2)
                ax.plot(gyX, gyY, 'w-', linewidth=2)

                if numbering:
                    pad = np.sum(self.hx)*0.04
                    for iy in range(int(nY)):
                        for ix in range(int(nX)):
                            iz = ix + iy*nX
                            if iz < self.nCz:
                                ax.text((ix+1)*self.vectorNx[-1]-pad,(iy)*self.vectorNy[-1]+pad,
                                         '#%i'%iz,color='w',verticalalignment='bottom',horizontalalignment='right',size='x-large')

        plt.show()
        return ph

    def plotGrid(self):
        """Plot the nodal, cell-centered and staggered grids for 1,2 and 3 dimensions."""
        if self.dim == 1:
            fig = plt.figure(1)
            fig.clf()
            ax = plt.subplot(111)
            xn = self.gridN
            xc = self.gridCC
            ax.hold(True)
            ax.plot(xn, np.ones(np.shape(xn)), 'bs')
            ax.plot(xc, np.ones(np.shape(xc)), 'ro')
            ax.plot(xn, np.ones(np.shape(xn)), 'k--')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            plt.show()
        elif self.dim == 2:
            fig = plt.figure(2)
            fig.clf()
            ax = plt.subplot(111)
            xn = self.gridN
            xc = self.gridCC
            xs1 = self.gridFx
            xs2 = self.gridFy

            ax.hold(True)
            ax.plot(xn[:, 0], xn[:, 1], 'bs')
            ax.plot(xc[:, 0], xc[:, 1], 'ro')
            ax.plot(xs1[:, 0], xs1[:, 1], 'g>')
            ax.plot(xs2[:, 0], xs2[:, 1], 'g^')
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            plt.show()
        elif self.dim == 3:
            fig = plt.figure(3)
            fig.clf()
            ax = fig.add_subplot(111, projection='3d')
            xn = self.gridN
            xc = self.gridCC
            xfs1 = self.gridFx
            xfs2 = self.gridFy
            xfs3 = self.gridFz

            xes1 = self.gridEx
            xes2 = self.gridEy
            xes3 = self.gridEz

            ax.hold(True)
            ax.plot(xn[:, 0], xn[:, 1], 'bs', zs=xn[:, 2])
            ax.plot(xc[:, 0], xc[:, 1], 'ro', zs=xc[:, 2])
            ax.plot(xfs1[:, 0], xfs1[:, 1], 'g>', zs=xfs1[:, 2])
            ax.plot(xfs2[:, 0], xfs2[:, 1], 'g<', zs=xfs2[:, 2])
            ax.plot(xfs3[:, 0], xfs3[:, 1], 'g^', zs=xfs3[:, 2])
            ax.plot(xes1[:, 0], xes1[:, 1], 'k>', zs=xes1[:, 2])
            ax.plot(xes2[:, 0], xes2[:, 1], 'k<', zs=xes2[:, 2])
            ax.plot(xes3[:, 0], xes3[:, 1], 'k^', zs=xes3[:, 2])
            ax.grid(True)
            ax.hold(False)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('x3')
            plt.show()
