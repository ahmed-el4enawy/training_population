import numpy as np
import time

import os
from sklearn.preprocessing import RobustScaler
try:
    from sklearn.preprocessing._data import _handle_zeros_in_scale
    print("_handle_zeros_in_scale exists")
except ImportError:
    print("_handle_zeros_in_scale NOT found")

filename = "E:\\T1D_population_training\\mock_scaler_memmap.dat"
shape = (20000000, 10)  # ~800 MB, small enough to test if memory shoots up by 800MB or stays tiny
mmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=shape)
mmap[:] = np.random.randn(*shape).astype(np.float32)
mmap.flush()




scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
t0 = time.time()
try:
    scaler.fit(mmap)
    
    print("Fit completed in", time.time() - t0, "s")
except Exception as e:
    print("Error during fit:", e)

del mmap
if os.path.exists(filename):
    os.remove(filename)
