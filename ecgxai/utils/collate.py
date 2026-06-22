import numpy as np
from typing import List, Union, Callable, Optional, Dict, Any, Tuple
import torch
from torch.utils.data._utils.collate import default_collate


def collate_time(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function.

    - Uses torch.utils.data.default_collate for all keys except `time`
    - If `time` is a np.ndarray of np.datetime64:
        returns np.ndarray(shape=(B,), dtype=object)
        where each entry is a list[str] of ISO timestamps
    """
    if not batch:
        raise ValueError("Empty batch passed to collate_time")

    if "time" not in batch[0]:
        return default_collate(batch)

    # Separate time from other fields
    time_values = [sample["time"] for sample in batch]
    rest = [{k: v for k, v in sample.items() if k != "time"} for sample in batch]

    out = default_collate(rest)

    v0 = time_values[0]

    if isinstance(v0, np.ndarray) and np.issubdtype(v0.dtype, np.datetime64):
        time_out: List[List[str]] = []

        for v in time_values:
            if not isinstance(v, np.ndarray):
                raise TypeError("Inconsistent `time` types within batch")
            ts = v.astype("datetime64[s]").astype(str).tolist()
            time_out.append(ts)

        out["time"] = np.asarray(time_out, dtype=object)
    else:
        out["time"] = default_collate(time_values)

    return out


def umcu_median_collate(
    batch: List[Dict[str, Any]],
    pad_label: Optional[str] = None,
) -> Dict[str, Any]:

    if not batch:
        raise ValueError("Empty batch")

    B = len(batch)

    def to_numpy(v: Any) -> np.ndarray:
        return v.detach().cpu().numpy() if torch.is_tensor(v) else v

    def collate(values: List[Any], key: str = "<unknown>") -> Any:
        non_null = [v for v in values if v is not None]
        if not non_null:
            return None

        numeric = [
            to_numpy(v)
            for v in non_null
            if isinstance(v, (np.ndarray, torch.Tensor))
        ]

        if numeric:
            non_empty_numeric = [v for v in numeric if v.size > 0]
            ref = non_empty_numeric[0] if non_empty_numeric else numeric[0]
            ndim = ref.ndim

            all_empty = all(
                (v.size == 0) if v.ndim > 0 else False
                for v in numeric
            )

            # -------- scalar --------
            if ndim == 0:
                if ref.dtype.kind == "f":
                    out = np.full((B, 1), np.nan, dtype=np.float64)
                elif ref.dtype.kind == "i":
                    out = np.full((B, 1), -1, dtype=np.int64)
                elif ref.dtype.kind == "b":
                    out = np.zeros((B, 1), dtype=np.bool_)
                else:
                    return default_collate(values)

                for i, v in enumerate(values):
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        out[i, 0] = to_numpy(v).item()

                return torch.from_numpy(out)

            # -------- 1D --------
            if ndim == 1:
                N = 1 if all_empty else max(1, max(v.size for v in numeric))

                if ref.dtype.kind == "f":
                    out = np.full((B, N), np.nan, dtype=np.float64)
                elif ref.dtype.kind == "i":
                    out = np.full((B, N), -1, dtype=np.int64)
                elif ref.dtype.kind == "b":
                    out = np.zeros((B, N), dtype=np.bool_)
                else:
                    return default_collate(values)

                for i, v in enumerate(values):
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        vv = to_numpy(v)
                        if vv.size > 0:
                            out[i, : vv.size] = vv

                return torch.from_numpy(out)

            # -------- 2D --------
            if ndim == 2:
                C = ref.shape[0]
                T = 1 if all_empty else max(v.shape[1] for v in numeric)

                if ref.dtype.kind == "f":
                    out = np.full((B, C, T), np.nan, dtype=np.float64)
                elif ref.dtype.kind == "i":
                    out = np.full((B, C, T), -1, dtype=np.int64)
                elif ref.dtype.kind == "b":
                    out = np.zeros((B, C, T), dtype=np.bool_)
                else:
                    return default_collate(values)

                for i, v in enumerate(values):
                    if isinstance(v, (np.ndarray, torch.Tensor)):
                        vv = to_numpy(v)
                        if vv.size > 0:
                            out[i, :, : vv.shape[1]] = vv

                return torch.from_numpy(out)

            raise ValueError(f"Unsupported numeric ndim={ndim}")

        # =====================================================
        # CATEGORICAL
        # =====================================================
        if all(
            isinstance(v, dict) and {"codes", "map", "missing"} <= v.keys()
            for v in non_null
        ):
            all_labels = []
            for v in non_null:
                if v["map"] is not None:
                    all_labels.extend(list(map(str, v["map"])))

            PAD = -1

            if not all_labels:
                out = np.full((B, 1), PAD, dtype=np.int64)

                missing = next(
                    (
                        v.get("missing", pad_label)
                        for v in non_null
                        if v.get("missing") is not None
                    ),
                    pad_label,
                )

                return {
                    "codes": torch.from_numpy(out),
                    "map": np.array([], dtype=str),
                    "missing": missing,
                }

            vocab = np.unique(np.asarray(all_labels, dtype=str))
            label_to_idx = {label: i for i, label in enumerate(vocab)}

            lengths = [
                len(to_numpy(v["codes"]))
                if to_numpy(v["codes"]).ndim
                else 1
                for v in non_null
            ]

            N = max(1, max(lengths))
            out = np.full((B, N), PAD, dtype=np.int64)

            for i, v in enumerate(values):
                if not isinstance(v, dict):
                    continue

                codes = to_numpy(v["codes"])
                src_map = v.get("map")
                if src_map is None:
                    continue

                src_map = np.asarray(src_map, dtype=str)

                if codes.ndim == 0:
                    out[i, 0] = label_to_idx[src_map[int(codes)]]
                else:
                    for j, c in enumerate(codes[:N]):
                        out[i, j] = label_to_idx[src_map[int(c)]]

            missing = next(
                (
                    v.get("missing", pad_label)
                    for v in non_null
                    if v.get("missing") is not None
                ),
                pad_label,
            )

            return {
                "codes": torch.from_numpy(out),
                "map": vocab,
                "missing": missing,
            }

        if all(isinstance(v, dict) for v in non_null):
            keys = set().union(*(v.keys() for v in non_null))
            return {
                k: collate([v.get(k) if isinstance(v, dict) else None for v in values], key=f"{key}.{k}")
                for k in keys
            }

        try:
            return default_collate(values)
        except Exception:
            return values

    keys = set().union(*(sample.keys() for sample in batch))
    result = {}
    for k in keys:
        try:
            result[k] = collate([sample.get(k) for sample in batch], key=k)
        except Exception as e:
            # Gather test IDs for context - try common ID field names
            test_ids = []
            for sample in batch:
                for id_field in ("test_id", "testid", "id", "sample_id", "ecg_id"):
                    if id_field in sample:
                        test_ids.append(sample[id_field])
                        break
                else:
                    test_ids.append("<no-id>")

            # Gather per-sample debug info for this key
            sample_info = []
            for idx, sample in enumerate(batch):
                val = sample.get(k)
                if val is None:
                    sample_info.append(f"  [{idx}] (testid={test_ids[idx]}): None")
                elif isinstance(val, (np.ndarray, torch.Tensor)):
                    arr = val.detach().cpu().numpy() if torch.is_tensor(val) else val
                    sample_info.append(
                        f"  [{idx}] (testid={test_ids[idx]}): shape={arr.shape} dtype={arr.dtype}"
                    )
                else:
                    sample_info.append(
                        f"  [{idx}] (testid={test_ids[idx]}): type={type(val).__name__} value={repr(val)[:80]}"
                    )

            raise type(e)(
                f"\nCollation failed for key='{k}'\n"
                f"Batch size: {B}\n"
                f"Per-sample values:\n" + "\n".join(sample_info) + f"\n\nOriginal error: {e}"
            ) from e

    return result

def umcu_median_collate_with_time(
    batch: List[Dict[str, Any]],
    pad_label: Optional[str] = None,
) -> Dict[str, Any]:

    if not batch:
        raise ValueError("Empty batch")

    if "time" not in batch[0]:
        return umcu_median_collate(batch, pad_label=pad_label)

    # ---------------- Separate time ----------------
    time_values = [sample["time"] for sample in batch]
    rest = [{k: v for k, v in sample.items() if k != "time"} for sample in batch]

    out = umcu_median_collate(rest, pad_label=pad_label)

    # ---------------- Process time ----------------
    v0 = time_values[0]

    if isinstance(v0, np.ndarray) and np.issubdtype(v0.dtype, np.datetime64):

        time_out: List[List[str]] = []

        for v in time_values:
            if not isinstance(v, np.ndarray):
                raise TypeError("Inconsistent `time` types within batch")

            ts = v.astype("datetime64[s]").astype(str).tolist()
            time_out.append(ts)

        out["time"] = np.asarray(time_out, dtype=object)

    else:
        out["time"] = default_collate(time_values)

    return out


def select_case_from_batch(
    batch_segmentation: Dict[str, Any],
    idx: int,
) -> Dict[str, Any]:

    noise_keys = {"noise_df", "secondary_noise_df"}
    meta_keys = {"meta", "secondary_meta"}

    def _select_tensor(v: torch.Tensor) -> np.ndarray:
        return v[idx].detach().cpu().numpy()

    out: Dict[str, Any] = {}

    for key, value in batch_segmentation.items():

        # ---------------- META ----------------
        if key in meta_keys and isinstance(value, dict):
            out[key] = {
                k: _select_tensor(v) if torch.is_tensor(v) else v
                for k, v in value.items()
            }
            continue

        # ---------------- BLOCK ----------------
        if isinstance(value, dict):

            block: Dict[str, Any] = {}

            for col, col_val in value.items():

                # -------- CATEGORICAL (decode only) --------
                if (
                    isinstance(col_val, dict)
                    and col_val.get("codes") is not None
                ):
                    codes = col_val["codes"][idx]
                    vocab = col_val.get("map")
                    missing = col_val.get("missing")

                    codes = (
                        codes.detach().cpu().tolist()
                        if torch.is_tensor(codes)
                        else list(codes)
                    )

                    decoded = [
                        (
                            vocab[c]
                            if vocab is not None and 0 <= int(c) < len(vocab)
                            else missing
                        )
                        for c in codes
                    ]

                    block[col] = decoded
                    continue

                # -------- NUMERIC (keep padding) --------
                if torch.is_tensor(col_val):
                    block[col] = _select_tensor(col_val)
                    continue

                # -------- RECURSIVE --------
                if isinstance(col_val, dict):
                    block[col] = {
                        sk: (
                            _select_tensor(sv)
                            if torch.is_tensor(sv)
                            else sv
                        )
                        for sk, sv in col_val.items()
                    }
                    continue

                block[col] = col_val

            out[key] = block
            continue

        # ---------------- TOP LEVEL ----------------
        if torch.is_tensor(value):
            out[key] = _select_tensor(value)
        elif isinstance(value, (list, tuple)) and len(value) > idx:
            out[key] = value[idx]
        else:
            out[key] = value

    return out


def select_case_from_sample(sample_segmentation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single-sample segmentation output (already unbatched),
    converting tensors to numpy / python objects and decoding categoricals.
    """

    noise_keys = {"noise_df", "secondary_noise_df"}
    meta_keys = {"meta", "secondary_meta"}

    def _is_all_nan(arr: np.ndarray) -> bool:
        return arr.size == 0 or np.all(np.isnan(arr))

    def _to_numpy_or_scalar(v: Any) -> Any:
        if torch.is_tensor(v):
            v = v.detach().cpu()
            return v.item() if v.ndim == 0 else v.numpy()
        return v

    out: Dict[str, Any] = {}

    for key, value in sample_segmentation.items():

        # ---- meta blocks ----
        if key in meta_keys and isinstance(value, dict):
            out[key] = {k: _to_numpy_or_scalar(v) for k, v in value.items()}
            continue

        # ---- structured blocks ----
        if isinstance(value, dict):
            block: Dict[str, Any] = {}

            for col, col_val in value.items():

                # ---- categorical ----
                if isinstance(col_val, dict) and col_val.get("codes") is not None:
                    codes = col_val["codes"]
                    vocab = col_val.get("map")
                    missing = col_val.get("missing")

                    if torch.is_tensor(codes):
                        codes = codes.detach().cpu().tolist()
                    else:
                        codes = list(codes)

                    decoded = [
                        (
                            vocab[c]
                            if vocab is not None and 0 <= int(c) < len(vocab)
                            else missing
                        )
                        for c in codes
                    ]

                    if key in noise_keys and all(
                        v == missing or v is None for v in decoded
                    ):
                        block[col] = []
                    else:
                        block[col] = decoded
                    continue

                # ---- numeric tensors ----
                if torch.is_tensor(col_val):
                    arr = col_val.detach().cpu().numpy()
                    block[col] = [] if key in noise_keys and _is_all_nan(arr) else arr
                    continue

                # ---- recursive dict ----
                if isinstance(col_val, dict):
                    sub = {}
                    for sk, sv in col_val.items():
                        sel = _to_numpy_or_scalar(sv)
                        if (
                            key in noise_keys
                            and isinstance(sel, np.ndarray)
                            and _is_all_nan(sel)
                        ):
                            sub[sk] = []
                        else:
                            sub[sk] = sel
                    block[col] = sub
                    continue

                block[col] = col_val

            out[key] = block
            continue

        # ---- top-level tensors ----
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().numpy()
            continue

        out[key] = value

    return out
