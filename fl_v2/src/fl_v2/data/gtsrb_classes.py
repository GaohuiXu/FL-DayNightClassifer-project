"""GTSRB 43-class index → name mapping.

Centralised lookup so analysis scripts and docs do not have to scrape
external sources. The names follow the canonical
`German Traffic Sign Recognition Benchmark` labels used in the
torchvision GTSRB loader and Stallkamp et al. 2012.

Reference: https://benchmark.ini.rub.de/gtsrb_dataset.html

The 43 classes split roughly into:

- Speed limit signs   (0-8): 20 / 30 / 50 / 60 / 70 / 80 / "End 80" /
                              100 / 120
- Prohibitory signs   (9-12, 15-17): No-passing × 2, Right-of-way,
                                      Priority road, No vehicles, Trucks
                                      no-passing, No entry
- Mandatory signs     (13-14): Yield, Stop
- Warning / triangular (18-31): General caution, curves, road work,
                                 etc.
- Direction signs     (32-40): End-of-restriction, Turn arrows,
                                Roundabout, etc.
- Pedestrian / school (27-29): Pedestrians, Children crossing, Bicycle

Use these for documentation, plot labels, and risk-audit narratives.
DO NOT use for filtering / partitioning logic — those should refer to
class indices (0-42) directly.
"""

GTSRB_CLASS_NAMES: dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice / snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

# Convenience: classes that are particularly safety-critical for AD
# perception. These are the targets a thesis-relevant backdoor attack
# should attack (vs. the current default class 2 = "Speed limit (50km/h)",
# which is the most-represented class and the easiest target — see
# cycle_02_codebase_risk_audit.md C3).
SAFETY_CRITICAL_CLASS_INDICES: tuple[int, ...] = (
    13,  # Yield
    14,  # Stop
    17,  # No entry
    27,  # Pedestrians
    28,  # Children crossing
)

# Cycle-02 poison-data-regime source-class pools (see the threat model in
# docs/roadmap/cycle_02_gradient_space_mechanism_study.md). `base` poisons
# common, well-represented source classes — the easy control where the
# backdoor competes with a strong clean gradient. `edge` poisons rare
# long-tail source classes — the durable, stealthy, AD-relevant case
# (cf. Neurotoxin / "Attack of the Tails"). The backdoor target stays
# class 14 (Stop) regardless of regime.
BASE_REGIME_SOURCE_CLASSES: tuple[int, ...] = (1, 2, 5, 12, 13)
EDGE_REGIME_SOURCE_CLASSES: tuple[int, ...] = (0, 19, 37)


def class_name(idx: int) -> str:
    """Return the human-readable name for a GTSRB class index 0-42."""
    if idx not in GTSRB_CLASS_NAMES:
        raise ValueError(
            f"GTSRB class index out of range: {idx}. Valid range is 0-42."
        )
    return GTSRB_CLASS_NAMES[idx]
