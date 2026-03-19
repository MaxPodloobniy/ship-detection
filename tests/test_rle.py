import numpy as np
import pytest

from src.utils import rle_decode, rle_encode


# ── decode ────────────────────────────────────────────────────────────


class TestRleDecode:
    def test_empty_string_returns_zeros(self):
        mask = rle_decode("")
        assert mask.shape == (768, 768)
        assert mask.sum() == 0

    def test_none_returns_zeros(self):
        mask = rle_decode(None)
        assert mask.shape == (768, 768)
        assert mask.sum() == 0

    def test_single_run(self):
        """First 3 pixels in column-major order → first 3 rows of column 0."""
        mask = rle_decode("1 3", shape=(4, 4))
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[0, 0] = 1
        expected[1, 0] = 1
        expected[2, 0] = 1
        np.testing.assert_array_equal(mask, expected)

    def test_multiple_runs(self):
        mask = rle_decode("1 2 5 3", shape=(4, 4))
        expected = np.zeros((4, 4), dtype=np.uint8)
        # run 1: pixels 1-2 (0-indexed: 0,1) → rows 0,1 col 0
        expected[0, 0] = 1
        expected[1, 0] = 1
        # run 2: pixels 5-7 (0-indexed: 4,5,6) → row 0 col 1, row 1 col 1, row 2 col 1
        expected[0, 1] = 1
        expected[1, 1] = 1
        expected[2, 1] = 1
        np.testing.assert_array_equal(mask, expected)

    def test_custom_shape(self):
        mask = rle_decode("1 1", shape=(3, 5))
        assert mask.shape == (3, 5)
        assert mask.sum() == 1
        assert mask[0, 0] == 1

    def test_full_mask(self):
        """All pixels set → entire mask is ones."""
        n = 8 * 8
        mask = rle_decode(f"1 {n}", shape=(8, 8))
        np.testing.assert_array_equal(mask, np.ones((8, 8), dtype=np.uint8))

    def test_odd_token_count_raises(self):
        with pytest.raises(ValueError, match="even number"):
            rle_decode("1 2 3")

    def test_column_major_order(self):
        """Pixel 5 in a 4×4 grid (1-indexed) should be row 0, col 1."""
        mask = rle_decode("5 1", shape=(4, 4))
        assert mask[0, 1] == 1
        assert mask.sum() == 1


# ── encode ────────────────────────────────────────────────────────────


class TestRleEncode:
    def test_empty_mask_returns_empty_string(self):
        mask = np.zeros((768, 768), dtype=np.uint8)
        assert rle_encode(mask) == ""

    def test_single_pixel(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        assert rle_encode(mask) == "1 1"

    def test_single_run(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 0] = 1
        mask[2, 0] = 1
        assert rle_encode(mask) == "1 3"

    def test_multiple_runs(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 0] = 1
        mask[0, 1] = 1
        mask[1, 1] = 1
        mask[2, 1] = 1
        assert rle_encode(mask) == "1 2 5 3"

    def test_full_mask(self):
        mask = np.ones((8, 8), dtype=np.uint8)
        assert rle_encode(mask) == "1 64"


# ── roundtrip ─────────────────────────────────────────────────────────


class TestRoundtrip:
    def test_encode_then_decode(self):
        """Encoding a mask and then decoding should return the original."""
        rng = np.random.RandomState(42)
        mask = (rng.rand(768, 768) > 0.95).astype(np.uint8)
        rle = rle_encode(mask)
        restored = rle_decode(rle, shape=(768, 768))
        np.testing.assert_array_equal(mask, restored)

    def test_decode_then_encode(self):
        rle_original = "1 5 10 3 100 20"
        mask = rle_decode(rle_original, shape=(768, 768))
        rle_result = rle_encode(mask)
        assert rle_result == rle_original

    def test_roundtrip_small_grid(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1, 0] = 1
        mask[2, 0] = 1
        mask[4, 2] = 1
        mask[5, 2] = 1

        rle = rle_encode(mask)
        restored = rle_decode(rle, shape=(6, 6))
        np.testing.assert_array_equal(mask, restored)

    def test_roundtrip_empty(self):
        mask = np.zeros((768, 768), dtype=np.uint8)
        rle = rle_encode(mask)
        restored = rle_decode(rle, shape=(768, 768))
        np.testing.assert_array_equal(mask, restored)
