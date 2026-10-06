import unittest
import numpy as np
from sklearn.preprocessing import RobustScaler

# Mock import from production
from train_population_model import fit_robust_scaler_memmap
from packed_population_model import PackedPopulationModel
from t1dsim_ai.population_model import CGMOHSUSimStateSpaceModel_V2
from t1dsim_ai.options import n_neurons_pop
import torch

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

    def test_packed_model_matches_official_and_exports(self):
        torch.manual_seed(123)
        official = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        packed = PackedPopulationModel(official, n_neurons_pop)

        x = torch.randn(32, 10)
        u = torch.randn(32, 2)

        with torch.no_grad():
            y_official = official(x, u)
            y_packed = packed(x, u)

        torch.testing.assert_close(y_packed, y_official, rtol=1e-5, atol=1e-6)
        self.assertEqual(packed.active_parameter_count(), sum(p.numel() for p in official.parameters()))

        exported = packed.export_official_state_dict()
        clone = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        clone.load_state_dict(exported)

        with torch.no_grad():
            y_clone = clone(x, u)
        torch.testing.assert_close(y_clone, y_packed, rtol=1e-5, atol=1e-6)

    def test_packed_model_one_adam_step_tracks_official(self):
        torch.manual_seed(456)
        official = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        packed = PackedPopulationModel(official, n_neurons_pop)

        x = torch.randn(16, 10)
        u = torch.randn(16, 2)
        target = torch.randn(16, 10)

        opt_official = torch.optim.Adam(official.parameters(), lr=1e-3)
        opt_packed = torch.optim.Adam(packed.parameters(), lr=1e-3)

        opt_official.zero_grad()
        loss_official = ((official(x, u) - target) ** 2).mean()
        loss_official.backward()
        opt_official.step()

        opt_packed.zero_grad()
        loss_packed = ((packed(x, u) - target) ** 2).mean()
        loss_packed.backward()
        opt_packed.step()

        packed.assert_inactive_zero()
        clone = CGMOHSUSimStateSpaceModel_V2(n_feat=n_neurons_pop)
        clone.load_state_dict(packed.export_official_state_dict())

        with torch.no_grad():
            y_official = official(x, u)
            y_clone = clone(x, u)

        torch.testing.assert_close(y_clone, y_official, rtol=2e-4, atol=2e-6)

if __name__ == '__main__':
    unittest.main()
