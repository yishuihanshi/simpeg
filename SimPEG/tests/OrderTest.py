import sys
sys.path.append('../../')
from SimPEG import TensorMesh, utils, LogicallyOrthogonalMesh
import numpy as np
import unittest


class OrderTest(unittest.TestCase):
    """

    OrderTest is a base class for testing convergence orders with respect to mesh
    sizes of integral/differential operators.

    Mathematical Problem:

        Given are an operator A and its discretization A[h]. For a given test function f
        and h --> 0  we compare:

        error(h) = \| A[h](f) - A(f) \|_{\infty}

        Note that you can provide any norm.

        Test is passed when estimated rate order of convergence is  at least within the specified tolerance of the
        estimated rate supplied by the user.

    Minimal example for a curl operator:

    class TestCURL(OrderTest):
        name = "Curl"

        def getError(self):
            # For given Mesh, generate A[h], f and A(f) and return norm of error.


            fun  = lambda x: np.cos(x)  # i (cos(y)) + j (cos(z)) + k (cos(x))
            sol = lambda x: np.sin(x)  # i (sin(z)) + j (sin(x)) + k (sin(y))


            Ex = fun(self.M.gridEx[:, 1])
            Ey = fun(self.M.gridEy[:, 2])
            Ez = fun(self.M.gridEz[:, 0])
            f = np.concatenate((Ex, Ey, Ez))

            Fx = sol(self.M.gridFx[:, 2])
            Fy = sol(self.M.gridFy[:, 0])
            Fz = sol(self.M.gridFz[:, 1])
            Af = np.concatenate((Fx, Fy, Fz))

            # Generate DIV matrix
            Ah = self.M.edgeCurl

            curlE = Ah*E
            err = np.linalg.norm((Ah*f -Af), np.inf)
            return err

        def test_order(self):
            # runs the test
            self.orderTest()

    See also: test_operatorOrder.py

    """

    name = "Order Test"
    expectedOrders = 2.  # This can be a list of orders, must be the same length as meshTypes
    _expectedOrder = 2.
    tolerance = 0.85
    meshSizes = [4, 8, 16, 32]
    meshTypes = ['uniformTensorMesh']
    _meshType = meshTypes[0]
    meshDimension = 3

    def setupMesh(self, nc):
        """
        For a given number of cells nc, generate a TensorMesh with uniform cells with edge length h=1/nc.
        """
        if 'TensorMesh' in self._meshType:
            if 'uniform' in self._meshType:
                h1 = np.ones(nc)/nc
                h2 = np.ones(nc)/nc
                h3 = np.ones(nc)/nc
                h = [h1, h2, h3]
            elif 'random' in self._meshType:
                h1 = np.random.rand(nc)
                h2 = np.random.rand(nc)
                h3 = np.random.rand(nc)
                h = [hi/np.sum(hi) for hi in [h1, h2, h3]]  # normalize
            else:
                raise Exception('Unexpected meshType')

            self.M = TensorMesh(h[:self.meshDimension])
            max_h = max([np.max(hi) for hi in self.M.h])
            return max_h

        elif 'LOM' in self._meshType:
            if 'uniform' in self._meshType:
                kwrd = 'rect'
            elif 'rotate' in self._meshType:
                kwrd = 'rotate'
            else:
                raise Exception('Unexpected meshType')
            if self.meshDimension == 2:
                X, Y = utils.exampleLomGird([nc, nc], kwrd)
                self.M = LogicallyOrthogonalMesh([X, Y])
            if self.meshDimension == 3:
                X, Y, Z = utils.exampleLomGird([nc, nc, nc], kwrd)
                self.M = LogicallyOrthogonalMesh([X, Y, Z])
            return 1./nc

    def getError(self):
        """For given h, generate A[h], f and A(f) and return norm of error."""
        return 1.

    def orderTest(self):
        """
        For number of cells specified in meshSizes setup mesh, call getError
        and prints mesh size, error, ratio between current and previous error,
        and estimated order of convergence.


        """
        assert type(self.meshTypes) == list, 'meshTypes must be a list'

        # if we just provide one expected order, repeat it for each mesh type
        if type(self.expectedOrders) == float or type(self.expectedOrders) == int:
            self.expectedOrders = [self.expectedOrders for i in self.meshTypes]

        assert type(self.expectedOrders) == list, 'expectedOrders must be a list'
        assert len(self.expectedOrders) == len(self.meshTypes), 'expectedOrders must have the same length as the meshTypes'

        for ii_meshType, meshType in enumerate(self.meshTypes):
            self._meshType = meshType
            self._expectedOrder = self.expectedOrders[ii_meshType]

            order = []
            err_old = 0.
            max_h_old = 0.
            for ii, nc in enumerate(self.meshSizes):
                max_h = self.setupMesh(nc)
                err = self.getError()
                if ii == 0:
                    print ''
                    print 'Testing convergence on ' + self.M._meshType + ' of:  ' + self.name
                    print '_____________________________________________'
                    print '   h  |    error    | e(i-1)/e(i) |  order'
                    print '~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~~~~|~~~~~~~~~~'
                    print '%4i  |  %8.2e   |' % (nc, err)
                else:
                    order.append(np.log(err/err_old)/np.log(max_h/max_h_old))
                    print '%4i  |  %8.2e   |   %6.4f    |  %6.4f' % (nc, err, err_old/err, order[-1])
                err_old = err
                max_h_old = max_h
            print '---------------------------------------------'
            passTest = np.mean(np.array(order)) > self.tolerance*self._expectedOrder
            if passTest:
                print ['The test be workin!', 'You get a gold star!', 'Yay passed!', 'Happy little convergence test!', 'That was easy!'][np.random.randint(5)]
            else:
                print 'Failed to pass test on ' + self._meshType + '.'
            print ''
            self.assertTrue(passTest)

if __name__ == '__main__':
    unittest.main()
