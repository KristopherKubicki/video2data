import os
import sys
import types

# Ensure utils2 can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "video2data"))
# Stub out cv2 and numpy for environments without these libraries installed
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
np_stub = types.SimpleNamespace(sum=sum)
sys.modules.setdefault("numpy", np_stub)
from utils2 import image


def test_chi2_distance_zero():
    hist = [0.1, 0.2, 0.3, 0.4]
    assert abs(image.chi2_distance(hist, hist)) < 1e-6


def test_chi2_distance_symmetry():
    hist_a = [0.1, 0.1, 0.4, 0.4]
    hist_b = [0.2, 0.3, 0.3, 0.2]
    val1 = image.chi2_distance(hist_a, hist_b)
    val2 = image.chi2_distance(hist_b, hist_a)
    assert abs(val1 - val2) < 1e-6

