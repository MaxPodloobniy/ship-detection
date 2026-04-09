import numpy as np


def rle_decode(mask_rle: str, shape: tuple[int, int] = (768, 768)) -> np.ndarray:
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

    return flat.reshape(shape, order="F")


def rle_encode(mask: np.ndarray) -> str:
    flat = mask.flatten(order="F").astype(np.uint8)

    if flat.max() == 0:
        return ""

    padded = np.concatenate([[0], flat, [0]])
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    starts = starts + 1
    lengths = ends - (starts - 1)

    tokens = []
    for start, length in zip(starts, lengths):
        tokens.append(str(start))
        tokens.append(str(length))

    return " ".join(tokens)
