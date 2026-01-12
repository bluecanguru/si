import unittest
import numpy as np
from si.neural_networks.optimizers import Adam

class TestAdam(unittest.TestCase):
    """
    Unit tests for the Adam optimizer.

    This class contains tests to verify the correct implementation of the Adam optimization
    algorithm, which is an adaptive learning rate optimization algorithm that combines
    the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
    The tests validate the weight update mechanism of the Adam optimizer.
    """

    def test_adam_update(self):
        """
        Test the weight update mechanism of the Adam optimizer.

        Verifies that the Adam optimizer correctly updates weights based on the first moment
        (mean) and second moment (uncentered variance) of the gradients. The test specifically
        checks the first update step (t=1) for the first component of the weight vector.

        The Adam update rule is defined as:
        1. Compute biased first moment estimate: m_t = β1*m_{t-1} + (1-β1)*g_t
        2. Compute biased second raw moment estimate: v_t = β2*v_{t-1} + (1-β2)*g_t^2
        3. Compute bias-corrected first moment estimate: m̂_t = m_t / (1-β1^t)
        4. Compute bias-corrected second raw moment estimate: v̂_t = v_t / (1-β2^t)
        5. Update parameters: θ_{t+1} = θ_t - η * m̂_t / (√v̂_t + ε)

        For the first update (t=1), m_0 and v_0 are initialized to 0, so the test verifies
        the correct computation of the first update step.
        """
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
