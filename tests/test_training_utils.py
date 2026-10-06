import unittest
import numpy as np
from sklearn.preprocessing import RobustScaler

# Mock import from production
from train_population_model import fit_robust_scaler_memmap

class TestTrainingUtils(unittest.TestCase):
    def test_custom_scaler_matches_sklearn(self):
        # Create a small synthetic dataset with known percentiles
        np.random.seed(42)
        # Using a dataset with some outliers to ensure IQR is correct
        data = np.random.randn(100, 3) * 10 + 50
        data[0, :] = [1000, 1000, 1000] # outliers
        data[1, :] = [-1000, -1000, -1000]
        
        # sklearn scaler
        sk_scaler = RobustScaler()
        sk_scaler.fit(data)
        
        # custom scaler
        cust_scaler = fit_robust_scaler_memmap(data)
        
        np.testing.assert_allclose(cust_scaler.center_, sk_scaler.center_)
        np.testing.assert_allclose(cust_scaler.scale_, sk_scaler.scale_)
        self.assertEqual(cust_scaler.n_features_in_, sk_scaler.n_features_in_)
        
    def test_zero_iqr_handling(self):
        # Test that features with 0 IQR don't cause divide by zero
        # Feature 0 has IQR = 0 (all values are 5)
        # Feature 1 has normal IQR
        data = np.zeros((100, 2))
        data[:, 0] = 5.0
        data[:, 1] = np.random.randn(100)
        
        cust_scaler = fit_robust_scaler_memmap(data)
        
        # If IQR is 0, sklearn sets scale to 1.0 to prevent division by zero
        self.assertEqual(cust_scaler.scale_[0], 1.0)
        self.assertEqual(cust_scaler.center_[0], 5.0)

if __name__ == '__main__':
    unittest.main()
