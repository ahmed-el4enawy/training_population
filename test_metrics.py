import numpy as np
from evaluate_population_model import window_outcomes_batch, paired_tost

def test_metrics():
    # 1. TIR, TAR, TBR
    # Synthetic BG bounds: 69, 70, 180, 181
    bg = np.array([[69.0, 70.0, 180.0, 181.0]])
    out = window_outcomes_batch(bg)
    
    # 70 to 180 inclusive -> 2 out of 4 = 50%
    assert np.isclose(out['TIR'][0], 50.0), f"TIR failed: {out['TIR'][0]}"
    # < 70 -> 1 out of 4 = 25%
    assert np.isclose(out['TBR'][0], 25.0), f"TBR failed: {out['TBR'][0]}"
    # > 180 -> 1 out of 4 = 25%
    assert np.isclose(out['TAR'][0], 25.0), f"TAR failed: {out['TAR'][0]}"
    
    # MG
    expected_mg = (69 + 70 + 180 + 181) / 4.0
    assert np.isclose(out['MG'][0], expected_mg), f"MG failed: {out['MG'][0]}"
    
    # 2. LBGI / HBGI
    # bg = [60, 100, 200, 300]
    # Hand-computed LBGI = 2.87, HBGI = 13.77
    bg_risk = np.array([[60.0, 100.0, 200.0, 300.0]])
    out_risk = window_outcomes_batch(bg_risk)
    assert np.isclose(out_risk['LBGI'][0], 2.87, atol=0.1), f"LBGI failed: {out_risk['LBGI'][0]}"
    assert np.isclose(out_risk['HBGI'][0], 13.77, atol=0.1), f"HBGI failed: {out_risk['HBGI'][0]}"

def test_tost():
    # 1. Clearly equivalent
    # diff = 0, se = 0, mean_diff < margin -> p_value = 0.0, Equivalent = True
    sim = np.array([100.0, 110.0, 120.0])
    act = np.array([100.0, 110.0, 120.0])
    res = paired_tost(sim, act, margin=5.0)
    assert res[5] is True, "Identical arrays failed equivalence"
    assert res[4] == 0.0, "p-value should be 0.0 for identical arrays"

    # 2. Clearly non-equivalent (identical arrays but mean_diff > margin)
    sim = np.array([110.0, 120.0, 130.0])
    act = np.array([100.0, 110.0, 120.0])
    # diff is 10 for all, mean_diff = 10. Margin = 5. Not equivalent.
    res = paired_tost(sim, act, margin=5.0)
    assert res[5] is False, "Arrays outside margin should not be equivalent"
    assert res[4] == 1.0, "p-value should be 1.0 for completely outside"
    
    # 3. Small n (< 2)
    sim = np.array([100.0])
    act = np.array([100.0])
    res = paired_tost(sim, act, margin=5.0)
    assert res[5] is False, "n < 2 should fail equivalence"

if __name__ == "__main__":
    test_metrics()
    test_tost()
    print("All tests passed.")
