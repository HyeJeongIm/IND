import os
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
from utils.utils import global_contrast_normalization
from collections import Counter
import json

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

def get_dataloaders(args):
    """Get dataloaders for the MVTec dataset."""
    data_dir = os.path.join(args.dataset_path, args.dataset_name)

    # Load min_max values from JSON file
    min_max = load_min_max(args.dataset_name, args.dataset_path)

    # min, max values for each class after applying GCN (as the original implementation)
    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[0]], [min_max[1] - min_max[0]])])

    # # Transformations applied to the images
    # transform = transforms.Compose([transforms.Resize((128, 128)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Lambda(lambda x: global_contrast_normalization(x)),
    #                                 # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #                                 transforms.Normalize([min_max[args.normal_class][0]],
    #                                                      [min_max[args.normal_class][1] - min_max[args.normal_class][0]])])

    """ Load train data (normal only) """
    train_img_paths = glob.glob(os.path.join(data_dir, 'train/good', '*.png')) # type = list
    train_labels = [0] * len(train_img_paths)  # All training data is normal (label=0)
    print(Counter(train_labels))

    """ Load test data """
    test_img_paths = []
    test_labels = []

    # Normal test data
    test_good_paths = glob.glob(os.path.join(data_dir, 'test/good', '*.png'))
    test_img_paths.extend(test_good_paths)
    test_labels.extend([0] * len(test_good_paths))  # Normal data (label=0)

    # Anomalous test data
    test_anomaly_dirs = [d for d in os.listdir(os.path.join(data_dir, 'test')) if d != 'good'] # ['poke', 'glue', 'fold', 'cut', 'color']
    for anomaly_dir in test_anomaly_dirs:
        anomaly_paths = glob.glob(os.path.join(data_dir, 'test', anomaly_dir, '*.png'))
        test_img_paths.extend(anomaly_paths)
        test_labels.extend([1] * len(anomaly_paths))  # Anomalous data (label=1)
    print(Counter(test_labels))
    
    # Create datasets
    train_dataset = CustomDatasetLoader(train_img_paths, train_labels, transform=transform)
    test_dataset = CustomDatasetLoader(test_img_paths, test_labels, transform=transform)

    # Create dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_test
