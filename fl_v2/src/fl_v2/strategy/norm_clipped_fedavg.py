from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg

from fl_v2.attacks_defenses import clip_updates_by_l2_norm
from fl_v2.strategy.norm_tracking_fedavg import (
    NormTrackingFedAvg,
    drop_nonfinite_replies,
    partition_sort_key,
)


class NormClippedFedAvg(NormTrackingFedAvg):
    """FedAvg with server-side L2 norm clipping and norm logging."""

    def __init__(self, clip_norm: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_norm = float(clip_norm)

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ):
        """Clip client updates, log norms, then aggregate via FedAvg."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        # NaN/Inf robustness — see drop_nonfinite_replies.
        valid_replies = drop_nonfinite_replies(valid_replies, self.arrayrecord_key)
        if not valid_replies:
            self._last_train_metrics = None
            return None, None

        # Deterministic ordering — see norm_tracking_fedavg.partition_sort_key.
        valid_replies.sort(key=partition_sort_key)

        if self.current_arrays is None:
            raise RuntimeError(
                "Current global arrays are not available. "
                "Make sure configure_train is called before aggregate_train."
            )

        global_params = self.current_arrays.to_numpy_ndarrays()

        # Keep original parameter names/order for each reply
        original_state_dicts = []
        client_params_list = []

        for msg in valid_replies:
            arrays = msg.content[self.arrayrecord_key]
            state_dict = arrays.to_torch_state_dict()
            original_state_dicts.append(state_dict)
            client_params_list.append(arrays.to_numpy_ndarrays())

        clipped_client_params_list, original_norms, clipped_norms = (
            clip_updates_by_l2_norm(
                global_params=global_params,
                client_params_list=client_params_list,
                clip_norm=self.clip_norm,
            )
        )

        # Replace client arrays with clipped arrays, preserving original keys
        for msg, clipped_params, state_dict in zip(
            valid_replies, clipped_client_params_list, original_state_dicts
        ):
            new_state_dict = OrderedDict()

            for (key, old_tensor), clipped_arr in zip(
                state_dict.items(), clipped_params
            ):
                clipped_arr = np.asarray(
                    clipped_arr, dtype=np.asarray(old_tensor.cpu()).dtype
                )
                clipped_arr = clipped_arr.reshape(tuple(old_tensor.shape))
                new_state_dict[key] = torch.tensor(
                    clipped_arr,
                    dtype=old_tensor.dtype,
                )

            msg.content[self.arrayrecord_key] = ArrayRecord(new_state_dict)

        # Log norms (both original and clipped) + gradient-space metrics
        # via parent's method. client_params_list here holds the
        # pre-clip updates, so the metrics describe the raw signature.
        self._compute_and_log_norms(
            server_round,
            original_norms,
            clipped_norms,
            global_params=global_params,
            client_params_list=client_params_list,
            partition_ids=[partition_sort_key(m) for m in valid_replies],
        )

        print(
            f"[Defense] round={server_round} norm clipping enabled "
            f"(clip_norm={self.clip_norm:.4f})",
            flush=True,
        )

        # Delegate aggregation to grandparent FedAvg
        # (skip NormTrackingFedAvg.aggregate_train to avoid recomputing norms)
        arrays, metrics = FedAvg.aggregate_train(self, server_round, valid_replies)
        self._last_train_metrics = self._capture_metrics(metrics)
        return arrays, metrics
