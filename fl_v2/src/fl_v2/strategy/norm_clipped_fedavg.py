from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from fl_v2.attacks_defenses import clip_updates_by_l2_norm


class NormClippedFedAvg(FedAvg):
    """FedAvg with server-side L2 norm clipping on client updates."""

    def __init__(self, clip_norm: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.clip_norm = float(clip_norm)
        self.current_arrays: ArrayRecord | None = None

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Store current global arrays before sending training instructions."""
        self.current_arrays = arrays.copy()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies,
    ):
        """Clip client updates before standard FedAvg aggregation."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

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

            for (key, old_tensor), clipped_arr in zip(state_dict.items(), clipped_params):
                clipped_arr = np.asarray(clipped_arr, dtype=np.asarray(old_tensor.cpu()).dtype)
                clipped_arr = clipped_arr.reshape(tuple(old_tensor.shape))
                new_state_dict[key] = torch.tensor(
                    clipped_arr,
                    dtype=old_tensor.dtype,
                )

            msg.content[self.arrayrecord_key] = ArrayRecord(new_state_dict)

        print(
            f"[Defense] round={server_round} norm clipping enabled "
            f"(clip_norm={self.clip_norm:.4f})",
            flush=True,
        )
        print(
            f"[Defense] original_norms={[round(x, 4) for x in original_norms]}",
            flush=True,
        )
        print(
            f"[Defense] clipped_norms={[round(x, 4) for x in clipped_norms]}",
            flush=True,
        )

        return super().aggregate_train(server_round, valid_replies)