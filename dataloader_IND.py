import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import json
from utils.utils import global_contrast_normalization

class CustomDatasetLoader(data.Dataset):
    """This class is needed to process batches for the dataloader."""
    def __init__(self, img_paths, labels, transform):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Return transformed items."""
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        """Number of samples."""
        return len(self.img_paths)

def load_min_max(dataset_name, dataset_path):
    """Load the min_max values from a JSON file."""
    min_max_path = os.path.join(dataset_path, 'min_max', f'{dataset_name}_min_max.json')
    with open(min_max_path, 'r') as f:
        min_max = json.load(f)
    return min_max[0]

def get_IND_loaders(args, normal_classes):
    """Get dataloader for the IND dataset with normal and anomalous data."""
    data_dir = os.path.join(args.dataset_test_path, args.testdataset_version, args.dataset_name)

    # Load min_max values from JSON file
    min_max = load_min_max(args.dataset_name, args.dataset_path)

    # Transformations applied to the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x)),
        transforms.Normalize([min_max[0]], [min_max[1] - min_max[0]])
    ])

    # Load test data
    test_img_paths = []
    test_labels = []

    # Normal test data (good and updated normal classes)
    for normal_class in normal_classes:
        normal_paths = glob.glob(os.path.join(data_dir, normal_class, '*.png'))
        test_img_paths.extend(normal_paths)
        test_labels.extend([0] * len(normal_paths))  # Normal data (label=0)

    # Anomalous test data
    # 여기에서 `normal_classes`에 속하지 않는 폴더들을 찾습니다.
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    test_anomaly_dirs = [d for d in all_dirs if d not in normal_classes]  # Anomalous directories

    for anomaly_dir in test_anomaly_dirs:
        anomaly_paths = glob.glob(os.path.join(data_dir, anomaly_dir, '*.png'))
        test_img_paths.extend(anomaly_paths)
        test_labels.extend([1] * len(anomaly_paths))  # Anomalous data (label=1)

    test_dataset = CustomDatasetLoader(test_img_paths, test_labels, transform=transform)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_test, dataloader_test


def get_only_IND_testloaders(args, new_instance):
    """Get dataloader for the IND dataset with only the specified abnormal instances."""
    data_dir = os.path.join(args.dataset_test_path, args.testdataset_version, args.dataset_name)

    # Load min_max values from JSON file
    min_max = load_min_max(args.dataset_name, args.dataset_path)

    # Transformations applied to the images
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x)),
        transforms.Normalize([min_max[0]], [min_max[1] - min_max[0]])
    ])

    # Load test data
    test_img_paths = []
    test_labels = []
    
    for abnormal_dir in new_instance:
        abnormal_paths = glob.glob(os.path.join(data_dir, abnormal_dir, '*.png'))
        test_img_paths.extend(abnormal_paths)
        test_labels.extend([1] * len(abnormal_paths))  # Anomalous data (label=1)

    test_dataset = CustomDatasetLoader(test_img_paths, test_labels, transform=transform)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_test, dataloader_test
