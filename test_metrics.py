import numpy as np
import scipy.stats as st
from evaluate_population_model import window_outcomes_batch, paired_tost

def test_metrics():
    # Synthetic BG: 1 window, 4 points: [60, 100, 200, 300]
    # n = 4
    # f_bg = 1.509 * (np.log(bg) ** 1.084 - 5.381)
    # f(60): 1.509 * (ln(60)^1.084 - 5.381) 
    #   ln(60) = 4.0943, 4.0943^1.084 = 4.673, 4.673-5.381 = -0.708 * 1.509 = -1.068
    #   r = 10 * f^2 = 11.41
    # f(100): ln(100) = 4.605, 4.605^1.084 = 5.321, 5.321-5.381 = -0.06 * 1.509 = -0.09
    #   r = 10 * (-0.09)^2 = 0.081
    # f(200): ln(200) = 5.298, 5.298^1.084 = 6.195, 6.195-5.381 = 0.814 * 1.509 = 1.228
    #   r = 10 * 1.228^2 = 15.08
    # f(300): ln(300) = 5.703, 5.703^1.084 = 6.711, 6.711-5.381 = 1.33 * 1.509 = 2.00
    #   r = 10 * 2.00^2 = 40.0
    
    # LBGI should be (11.41 + 0.081 + 0 + 0) / 4 = 2.87
    # HBGI should be (0 + 0 + 15.08 + 40.0) / 4 = 13.77
    bg = np.array([[60, 100, 200, 300]], dtype=np.float64)
    out = window_outcomes_batch(bg)
    print("Test Metrics:")
    print(f"LBGI: {out['LBGI'][0]:.2f}")
    print(f"HBGI: {out['HBGI'][0]:.2f}")
    
    # Paired TOST test
    sim = np.array([1, 2, 3, 4, 5])
    actual = np.array([1, 2, 3, 4, 5])
    # diff = 0, se = 1e-12, margin = 5, t = 5/1e-12 -> p_value = 0 (Equivalent)
    res = paired_tost(sim, actual, 5.0)
    print("TOST Test:")
    print(res)

if __name__ == "__main__":
    test_metrics()
