import unittest
import numpy as np
from si.neural_networks.optimizers import Adam

class TestAdam(unittest.TestCase):

    def test_adam_update(self):
        
        w = np.array([0.1, 0.2, 0.3])
        grad = np.array([0.01, 0.02, 0.03])

        lr = 0.1    
        b1 = 0.9    
        b2 = 0.999  
        eps = 1e-8  

        adam = Adam(learning_rate=lr, beta_1=b1, beta_2=b2, epsilon=eps)

        w_new = adam.update(w, grad)

        m1 = (1 - b1) * grad[0]
        v1 = (1 - b2) * (grad[0]**2)
        m_hat = m1 / (1 - b1**1)
        v_hat = v1 / (1 - b2**1)
        expected_w0 = w[0] - lr * (m_hat / (np.sqrt(v_hat) + eps))

        self.assertAlmostEqual(w_new[0], expected_w0,
                              msg="First component of updated weights should match expected value")

        self.assertEqual(adam.t, 1,
                        msg="Time step counter should be 1 after first update")

if __name__ == '__main__':
    unittest.main()
