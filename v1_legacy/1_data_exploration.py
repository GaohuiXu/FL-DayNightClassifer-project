import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

print("loading nuScenes-mini dataset...")
nusc = NuScenes(version='v1.0-mini', dataroot='./data', verbose=True)

print("--- preparing the description of 10 scenes ---")

for scene in nusc.scene:
    desc = scene['description'].lower()
    print(f"{scene['name']}: {desc}")

