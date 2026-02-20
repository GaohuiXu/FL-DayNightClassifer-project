import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from nuscenes.nuscenes import NuScenes

# ==========================================
# 1. Custom PyTorch Dataset
# ==========================================
class DayNightDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list: List of tuples containing (image_path, label).
            transform: Image transformations to apply.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ==========================================
# 2. Federated Data Partitioning & Loading
# ==========================================
def load_data(partition_id, num_partitions):
    print(f"Loading data for Client {partition_id}...")
    
    # Initialize nuScenes engine
    nusc = NuScenes(version='v1.0-mini', dataroot='./data', verbose=False)
    
    all_data = []
    
    # Extract frontal camera image paths and labels from all scenes
    for scene in nusc.scene:
        desc = scene['description'].lower()
        label = 1 if 'night' in desc else 0  # 1 for night, 0 for day
        
        current_sample_token = scene['first_sample_token']
        while current_sample_token != '':
            sample = nusc.get('sample', current_sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            
            img_path = os.path.join('./data', cam_front_data['filename'])
            all_data.append((img_path, label))
            
            current_sample_token = sample['next']
            
    # --- Data Partitioning (Sequential to simulate Non-IID) ---
    partition_size = len(all_data) // num_partitions
    start_idx = partition_id * partition_size
    # The last partition takes any remaining samples
    end_idx = start_idx + partition_size if partition_id != num_partitions - 1 else len(all_data)
    
    client_data = all_data[start_idx:end_idx]
    
    # --- Train/Val Split (80/20) ---
    train_size = int(0.8 * len(client_data))
    train_data = client_data[:train_size]
    val_data = client_data[train_size:]
    
    # Image transformations: Resize to 128x128, convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = DayNightDataset(train_data, transform=transform)
    valset = DayNightDataset(val_data, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32, shuffle=False)
    
    return trainloader, valloader

# ==========================================
# 3. Local Standalone Test
# ==========================================
if __name__ == "__main__":
    t_loader, v_loader = load_data(partition_id=0, num_partitions=2)
    
    print(f"\n--- Data loading test successful! ---")
    print(f"Number of training batches for Client 0: {len(t_loader)}")
    print(f"Number of validation batches for Client 0: {len(v_loader)}")
    
    for images, labels in t_loader:
        print(f"Batch image tensor shape: {images.shape}")  # Expected: [32, 3, 128, 128]
        print(f"Batch labels: {labels}")
        break