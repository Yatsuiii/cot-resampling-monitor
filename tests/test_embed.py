import numpy as np

from mats.embed import HashEmbedder, cosine_matrix


def test_encode_is_unit_norm():
    vecs = HashEmbedder().encode(["the cat sat", "a completely different phrase"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_identical_text_has_cosine_one():
    e = HashEmbedder()
    a = e.encode(["conclude with option C"])
    assert cosine_matrix(a, a)[0, 0] == 1.0


def test_dissimilar_text_has_low_cosine():
    e = HashEmbedder()
    a = e.encode(["the first choice follows from the evidence"])
    b = e.encode(["quantum entanglement between distant particles"])
    assert cosine_matrix(a, b)[0, 0] < 0.2


def test_empty_input_is_handled():
    assert HashEmbedder().encode([]).shape == (0, 64)
    assert cosine_matrix(np.zeros((0, 64)), np.zeros((3, 64))).shape == (0, 3)
