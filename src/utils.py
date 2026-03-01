import numpy as np


def rle_decode(mask_rle: str, shape: tuple[int, int] = (768, 768)) -> np.ndarray:
    """
    Decode a run-length encoded string into a binary mask.

    Kaggle Ship Detection Challenge uses column-major (Fortran) order:
    pixels go top-to-bottom first, then left-to-right.
    Pixel indexing is 1-based.

    Args:
        mask_rle: RLE string with space-separated pairs (start length start length ...).
                  Empty string or NaN means no mask (returns zeros).
        shape: (height, width) of the output mask. Default is (768, 768).

    Returns:
        np.ndarray: Binary mask of shape (height, width), dtype uint8.
    """
    if not isinstance(mask_rle, str) or mask_rle.strip() == "":
        return np.zeros(shape, dtype=np.uint8)

    tokens = list(map(int, mask_rle.split()))
    if len(tokens) % 2 != 0:
        raise ValueError(
            f"RLE string must contain an even number of values, got {len(tokens)}"
        )

    starts = tokens[0::2]
    lengths = tokens[1::2]

    flat = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        # Kaggle RLE is 1-indexed
        begin = start - 1
        flat[begin : begin + length] = 1

    # Reshape in Fortran (column-major) order
    return flat.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask into a run-length encoded string.

    Uses column-major (Fortran) order with 1-based indexing,
    matching the Kaggle Ship Detection Challenge format.

    Args:
        mask: Binary mask of shape (height, width).

    Returns:
        RLE string with space-separated pairs (start length ...),
        or empty string if the mask contains no positive pixels.
    """
    # Flatten in Fortran (column-major) order
    flat = mask.flatten(order="F").astype(np.uint8)

    if flat.max() == 0:
        return ""

    # Pad with zeros to detect transitions at boundaries
    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)

    # Starts: where diff == 1 (transition 0 -> 1)
    # Ends:   where diff == -1 (transition 1 -> 0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # Convert to 1-based indexing
    starts = starts + 1
    lengths = ends - (starts - 1)

    tokens = []
    for s, l in zip(starts, lengths):
        tokens.append(str(s))
        tokens.append(str(l))

    return " ".join(tokens)
