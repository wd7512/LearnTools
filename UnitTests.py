from learntools import Network
import unittest
import numpy as np
import os
from time import time
  
  
def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func


class TestNetwork(unittest.TestCase):

    @timer_func
    def test_relu1(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,1)
        lay1.weights[0] = 1
        lay1.biases[0] = -1
        test_network.add_layer(lay1)
        test_network.add_layer(Network.relu())

        test_network.reset()
        X = np.array([[1]])
        self.assertEqual(test_network.forward(X),0)
        

    @timer_func
    def test_relu2(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,1)
        lay1.weights[0] = 2
        lay1.biases[0] = 0
        test_network.add_layer(lay1)
        test_network.add_layer(Network.relu())

        X = np.array([[1]])
        self.assertEqual(test_network.forward(X),2)

    @timer_func
    def test_softmax(self):
        test_network = Network.network(1,2)
        lay1 = Network.layer_dense(1,2)
        lay1.weights = np.array([[1,2]])
        
        test_network.add_layer(lay1)
        test_network.add_layer(Network.softmax())
        
        X = np.array([[1],[2]])
        np.testing.assert_equal(test_network.forward(X), 
                                np.array([[np.exp(1)/(np.exp(1)+np.exp(2)),np.exp(2)/(np.exp(1)+np.exp(2))],
                                          [np.exp(2)/(np.exp(2)+np.exp(4)),np.exp(4)/(np.exp(2)+np.exp(4))]]))
        
    @timer_func
    def test_file_saving(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,10)
        lay1.weights = np.random.normal(size=np.shape(lay1.weights))
        lay2 = Network.layer_dense(10,11)
        lay1.weights = np.random.normal(size=np.shape(lay1.weights))
        test_network.add_layer(lay1)
        test_network.add_layer(Network.relu())
        test_network.add_layer(lay2)
        test_network.add_layer(Network.sigmoid())

        input = [np.random.normal(size=1)]
        output = test_network.forward(input)[0][0]

        test_network.save_to_file('UnitTest.pkl')
        del test_network
        new_net = Network.load_network_from_file('UnitTest.pkl')

        self.assertEqual(new_net.forward(input)[0][0],output)

        os.remove('UnitTest.pkl')
        
    @timer_func
    def test_one_to_one(self):
        test_network = Network.network(1,1)
        lay1 = Network.layer_dense(1,10)
        lay1.weights = np.ones(shape=np.shape(lay1.weights))
        lay2 = Network.layer_one_to_one(10)
        lay2.weights = np.ones(shape=np.shape(lay2.weights))
        lay3 = Network.layer_dense(10,1)
        lay3.weights = np.ones(shape=np.shape(lay3.weights))

        test_network.add_layer(lay1)
        test_network.add_layer(lay2)
        test_network.add_layer(lay3)

        X = np.array([[1]])

        output = test_network.forward(X)

        self.assertEqual(output,10)

    @timer_func
    def test_reset(self):
        test_network = Network.network(3,1)
        test_network.add_layer(Network.layer_dense(3,10))
        test_network.add_layer(Network.relu())
        test_network.add_layer(Network.layer_one_to_one(10))
        test_network.add_layer(Network.softmax())
        test_network.add_layer(Network.layer_dropout(10,0))
        test_network.add_layer(Network.layer_1dconv(10,5))
        test_network.add_layer(Network.sigmoid())
        test_network.add_layer(Network.layer_dense(10,1))

        test_network.reset()

        X = np.array([[1,1,1]])

        self.assertEqual(test_network.forward(X),0)

    @timer_func
    def test_1dconv(self):
        test_network = Network.network(6,6)
        test_network.add_layer(Network.layer_1dconv(6,3))

        X = np.array([[1,1,1,4,1,4],
                      [0,-1,0,-1,0,1]])

        Y_pred = test_network.forward(X)
        
        self.assertListEqual(list(Y_pred[0]),[1,1,2,2,3,2.5])
        self.assertListEqual(list(Y_pred[1]),[-1/2,-1/3,-2/3,-1/3,0,1/2])

if __name__ == '__main__':
    unittest.main()