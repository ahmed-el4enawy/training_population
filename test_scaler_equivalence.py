import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing._data import _handle_zeros_in_scale

def custom_fit_robust_scaler(array):
    n_rows, n_features = array.shape
    centers = np.zeros(n_features, dtype=np.float64)
    scales = np.zeros(n_features, dtype=np.float64)
    
    for i in range(n_features):
        col = array[:, i]
        # sklearn uses nanpercentile by default if not told otherwise, but we use percentile since no NaNs
        q25, median, q75 = np.percentile(col, [25.0, 50.0, 75.0])
        iqr = q75 - q25
        centers[i] = median
        scales[i] = iqr
        
    scales = _handle_zeros_in_scale(scales, copy=False)
    
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    scaler.center_ = centers
    scaler.scale_ = scales
    scaler.n_features_in_ = n_features
    return scaler

# Create deterministic dataset
np.random.seed(42)
X = np.random.randn(1000, 5).astype(np.float32)
# Add some zeros to force zero-scale test
X[:, 1] = 0.0
X[0, 1] = 1.0

# 1. Standard sklearn
sk_scaler = RobustScaler()
sk_scaler.fit(X)
X_sk = sk_scaler.transform(X)

# 2. Custom implementation
custom_scaler = custom_fit_robust_scaler(X)
X_custom = custom_scaler.transform(X)

print("Center equal:", np.allclose(sk_scaler.center_, custom_scaler.center_, atol=1e-6))
print("Scale equal:", np.allclose(sk_scaler.scale_, custom_scaler.scale_, atol=1e-6))
print("Transform equal:", np.allclose(X_sk, X_custom, atol=1e-6))
