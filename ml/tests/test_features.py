import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from affective_ml.features import BASIC_FEATURE_NAMES, basic_features_frame, compose_matrix


def test_basic_features_shape_and_values():
    series = pd.Series(["Produk BAGUS!!! sangat puas..", "jelek, minta refund"])
    frame = basic_features_frame(series)

    assert list(frame.columns) == list(BASIC_FEATURE_NAMES)
    assert frame.shape == (2, 11)
    assert frame.loc[0, "n_exclamation"] == 3
    assert frame.loc[0, "n_allcaps"] == 1
    assert frame.loc[0, "n_ellipsis"] == 1
    assert frame.loc[1, "n_demands"] == 1


def test_compose_matrix_appends_indicator():
    tfidf = csr_matrix(np.ones((3, 4)))
    basic = pd.DataFrame(np.zeros((3, 11)), columns=BASIC_FEATURE_NAMES)
    indicator = np.array([1, 0, 1])

    with_indicator = compose_matrix(tfidf, basic, indicator)
    without_indicator = compose_matrix(tfidf, basic)

    assert with_indicator.shape == (3, 16)
    assert without_indicator.shape == (3, 15)
    assert with_indicator[:, -1].toarray().ravel().tolist() == [1, 0, 1]
