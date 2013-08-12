import numpy as np
import unittest
from OrderTest import OrderTest


# MATLAB code:

# syms x y z

# ex = x.^2+y.*z;
# ey = (z.^2).*x+y.*z;
# ez = y.^2+x.*z;

# e = [ex;ey;ez];

# sigma1 = x.*y+1;
# sigma2 = x.*z+2;
# sigma3 = 3+z.*y;
# sigma4 = 0.1.*x.*y.*z;
# sigma5 = 0.2.*x.*y;
# sigma6 = 0.1.*z;

# S1 = [sigma1,0,0;0,sigma1,0;0,0,sigma1];
# S2 = [sigma1,0,0;0,sigma2,0;0,0,sigma3];
# S3 = [sigma1,sigma4,sigma5;sigma4,sigma2,sigma6;sigma5,sigma6,sigma3];

# i1 = int(int(int(e.'*S1*e,x,0,1),y,0,1),z,0,1);
# i2 = int(int(int(e.'*S2*e,x,0,1),y,0,1),z,0,1);
# i3 = int(int(int(e.'*S3*e,x,0,1),y,0,1),z,0,1);


class TestInnerProducts(OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['uniformTensorMesh', 'uniformLOM', 'rotateLOM']
    meshDimension = 3
    meshSizes = [16, 32]

    def getError(self):

        call = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])

        ex = lambda x, y, z: x**2+y*z
        ey = lambda x, y, z: (z**2)*x+y*z
        ez = lambda x, y, z: y**2+x*z

        sigma1 = lambda x, y, z: x*y+1
        sigma2 = lambda x, y, z: x*z+2
        sigma3 = lambda x, y, z: 3+z*y
        sigma4 = lambda x, y, z: 0.1*x*y*z
        sigma5 = lambda x, y, z: 0.2*x*y
        sigma6 = lambda x, y, z: 0.1*z

        Gc = self.M.gridCC
        if self.sigmaTest == 1:
            sigma = np.c_[call(sigma1, Gc)]
            analytic = 647./360  # Found using matlab symbolic toolbox.
        elif self.sigmaTest == 3:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 37./12  # Found using matlab symbolic toolbox.
        elif self.sigmaTest == 6:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc),
                          call(sigma4, Gc), call(sigma5, Gc), call(sigma6, Gc)]
            analytic = 69881./21600  # Found using matlab symbolic toolbox.

        if self.location == 'edges':
            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]
            Ec = np.vstack((cart(self.M.gridEx),
                            cart(self.M.gridEy),
                            cart(self.M.gridEz)))
            E = self.M.projectEdgeVector(Ec)
            A = self.M.getEdgeInnerProduct(sigma)
            numeric = E.T.dot(A.dot(E))
        elif self.location == 'faces':
            cart = lambda g: np.c_[call(ex, g), call(ey, g), call(ez, g)]
            Fc = np.vstack((cart(self.M.gridFx),
                            cart(self.M.gridFy),
                            cart(self.M.gridFz)))
            F = self.M.projectFaceVector(Fc)
            A = self.M.getFaceInnerProduct(sigma)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = 'edges'
        self.sigmaTest = 1
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = 'edges'
        self.sigmaTest = 3
        self.orderTest()

    def test_order6_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = 'edges'
        self.sigmaTest = 6
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = 'faces'
        self.sigmaTest = 1
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = 'faces'
        self.sigmaTest = 3
        self.orderTest()

    def test_order6_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = 'faces'
        self.sigmaTest = 6
        self.orderTest()


class TestInnerProducts2D(OrderTest):
    """Integrate an function over a unit cube domain using edgeInnerProducts and faceInnerProducts."""

    meshTypes = ['uniformTensorMesh', 'uniformLOM', 'rotateLOM']
    meshDimension = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):

        z = 5  # Because 5 is just such a great number.

        call = lambda fun, xy: fun(xy[:, 0], xy[:, 1])

        ex = lambda x, y: x**2+y*z
        ey = lambda x, y: (z**2)*x+y*z

        sigma1 = lambda x, y: x*y+1
        sigma2 = lambda x, y: x*z+2
        sigma3 = lambda x, y: 3+z*y

        Gc = self.M.gridCC
        if self.sigmaTest == 1:
            sigma = np.c_[call(sigma1, Gc)]
            analytic = 144877./360  # Found using matlab symbolic toolbox. z=5
        elif self.sigmaTest == 2:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc)]
            analytic = 189959./120  # Found using matlab symbolic toolbox. z=5
        elif self.sigmaTest == 3:
            sigma = np.c_[call(sigma1, Gc), call(sigma2, Gc), call(sigma3, Gc)]
            analytic = 781427./360  # Found using matlab symbolic toolbox. z=5

        if self.location == 'edges':
            cart = lambda g: np.c_[call(ex, g), call(ey, g)]
            Ec = np.vstack((cart(self.M.gridEx),
                            cart(self.M.gridEy)))
            E = self.M.projectEdgeVector(Ec)
            A = self.M.getEdgeInnerProduct(sigma)
            numeric = E.T.dot(A.dot(E))
        elif self.location == 'faces':
            cart = lambda g: np.c_[call(ex, g), call(ey, g)]
            Fc = np.vstack((cart(self.M.gridFx),
                            cart(self.M.gridFy)))
            F = self.M.projectFaceVector(Fc)
            A = self.M.getFaceInnerProduct(sigma)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "2D Edge Inner Product - Isotropic"
        self.location = 'edges'
        self.sigmaTest = 1
        self.orderTest()

    def test_order3_edges(self):
        self.name = "2D Edge Inner Product - Anisotropic"
        self.location = 'edges'
        self.sigmaTest = 2
        self.orderTest()

    def test_order6_edges(self):
        self.name = "2D Edge Inner Product - Full Tensor"
        self.location = 'edges'
        self.sigmaTest = 3
        self.orderTest()

    def test_order1_faces(self):
        self.name = "2D Face Inner Product - Isotropic"
        self.location = 'faces'
        self.sigmaTest = 1
        self.orderTest()

    def test_order2_faces(self):
        self.name = "2D Face Inner Product - Anisotropic"
        self.location = 'faces'
        self.sigmaTest = 2
        self.orderTest()

    def test_order3_faces(self):
        self.name = "2D Face Inner Product - Full Tensor"
        self.location = 'faces'
        self.sigmaTest = 3
        self.orderTest()


if __name__ == '__main__':
    unittest.main()
