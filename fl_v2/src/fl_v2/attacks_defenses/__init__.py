from fl_v2.attacks_defenses.attacks.label_flipping import (
    LabelFlippingDataset,
    parse_client_ids,
)
from fl_v2.attacks_defenses.attacks.model_replacement import (
    compute_replacement_scale,
    scale_client_update,
)
from fl_v2.attacks_defenses.attacks.pixel_backdoor import (
    PixelBackdoorDataset,
    dba_subregions,
    make_pixel_trigger_fn,
)
from fl_v2.attacks_defenses.attacks.neurotoxin import (
    compute_clean_proxy_gradient,
    project_grad_away_from_topk,
    topk_mask_from_proxy,
)
from fl_v2.attacks_defenses.defenses.norm_clipping import (
    clip_updates_by_l2_norm,
    compute_gradient_space_metrics,
    compute_update_norms,
)

__all__ = [
    "LabelFlippingDataset",
    "parse_client_ids",
    "PixelBackdoorDataset",
    "make_pixel_trigger_fn",
    "dba_subregions",
    "compute_clean_proxy_gradient",
    "topk_mask_from_proxy",
    "project_grad_away_from_topk",
    "compute_replacement_scale",
    "scale_client_update",
    "clip_updates_by_l2_norm",
    "compute_gradient_space_metrics",
    "compute_update_norms",
]