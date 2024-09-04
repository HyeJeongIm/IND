import numpy as np
import argparse 
import torch

from train import TrainerDeepSVDD
# from preprocess import get_mnist
from dataloader import get_dataloaders
from dataloader_IND import get_IND_loaders, load_min_max, CustomDatasetLoader, get_only_IND_testloaders
import os
import visualization
from test import eval, eval_single_image, eval, eval_single_image_IDPOC
from torchvision import transforms
from utils.utils import global_contrast_normalization
from torch.utils.data import DataLoader
import random
import ipdb
def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep SVDD Training on MVTecAD Datasets")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')  # 시드 추가
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=100, help="Number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6, help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3, help='Weight decay hyperparameter for the L2 regularization during autoencoder pretraining')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='Learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50], help='Milestones at which the scheduler multiplies the learning rate by 0.1')
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True, help='Whether to pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of the latent variable z')
    # Dataset
    parser.add_argument('--dataset_path', type=str, default='./data/MVTecAD', help='Path to the dataset root directory')
    # parser.add_argument('--dataset_name', type=str, default='leather', choices=['leather', 'metal_nut', 'wood'], help='Name of the dataset to use')
    parser.add_argument('--dataset_name', type=str, default='leather', choices=['leather'], help='Name of the dataset to use')
    parser.add_argument('--dataset_test_path', type=str, default='./data/IND', help='Path to the dataset root directory')
    parser.add_argument('--testdataset_version', type=str, default='v6', help='Version of the IND test dataset')

    # Save path
    parser.add_argument('--output_path', type=str, default='./results/', help='Path to save the output models and results')
    parser.add_argument('--normal_class', type=int, default=0, help='Class to be treated as normal. The rest will be considered as anomalous.') 
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    set_seed(args.seed)

    # normal class 
    normal_classes = ['good']  
    # novel classes
    class_names = ['leather', 'metal_nut', 'wood']

    # 각 novel class의 threshold 설정
    thresholds = {
        'leather': 102.45,
        'metal_nut': 36.15,
        'wood': 32.12
    }
    
    # new instance
    instance_folders = ['glue', 'color', 'cut', 'fold', 'poke']
    for instance in instance_folders:
        args.dataset_name = 'leather'
        print(f"Processing {instance} dataset...")
        
        novel_buffer = []
        known_buffer = [] 
        results = []

        new_instance = [instance]
        normal_classes.append(instance)

        # normal + novel(new instance) dataset
        data_NI = get_IND_loaders(args, normal_classes)  
        # only novel(new instance) dataset
        data_only_NI = get_only_IND_testloaders(args, new_instance) 
        _, data_only_NI_loader = data_only_NI
        
        for idx, (image, label) in enumerate(data_only_NI[1]):
            image = image.to(device)
            scores = {}
            is_novel = True  # 이미지가 모든 모델에서 novel로 분류되는지 여부

            for class_name in class_names:
                """ Create Model & Load weight """
                args.dataset_name = class_name
                model = TrainerDeepSVDD(args, data_NI, device=device) 
                state_dict = torch.load(model.trained_weights_path)
                # print(model.trained_weights_path)
                model.net.load_state_dict(state_dict['net_dict'])
                model.c = torch.Tensor(state_dict['center']).to(model.device)

                # score = eval_single_image_IDPOC(model.net, model.c, image, device)
                score = eval_single_image(model.net, model.c, image, device)
                scores[class_name] = score[0] 
    
                """ 
                    1) Known vs Novel 
                        - 모든 model에서 Novel로 판단한 경우만, Novel로 설정
                        - 다른 경우는 모두 Known으로 설정
                """
                # Score와 threshold 비교
                if scores[class_name] <= thresholds[class_name]:
                    is_novel = False

            # Novel or Known 버퍼에 추가
            if is_novel:
                novel_buffer.append((data_only_NI_loader.dataset.img_paths[idx], label))
                novelty_status = "Novel"
            else:
                known_buffer.append((data_only_NI_loader.dataset.img_paths[idx], label))
                novelty_status = "Known"

            results.append((idx, label[0].item(), scores, novelty_status))
        
        # Save results to file
        results_dir = os.path.join(args.output_path, "IND_buffer", args.testdataset_version, instance)
        os.makedirs(results_dir, exist_ok=True)  
   
        results_path = os.path.join(results_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            for idx, label, scores, novelty_status in results:
                min_class_score = min(scores.values())  
                f.write(f"Image {idx + 1} - Label: {label}, Scores: {scores}, Status: {novelty_status}, Min Score: {min_class_score:.2f}\n")

        print(f"Initial results saved to {results_path}")

        # Save novel images
        visualization.visualize_and_save_novel_images(novel_buffer, results_dir)
        
        """ 
            2) OOD update 
                - novel_buffer에 novel image가 있는 경우, update 진행

        """
        # Fine-tune if "Novel"
        if len(novel_buffer) > 0:
            print(f"Fine-tuning with {len(novel_buffer)} Novel images from {instance}")
            args.dataset_name = 'leather'
            min_max = load_min_max(args.dataset_name, args.dataset_path)

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: global_contrast_normalization(x)),
                transforms.Normalize([min_max[0]], [min_max[1] - min_max[0]])
            ])

            novel_images, novel_labels = zip(*novel_buffer)
            novel_dataset = CustomDatasetLoader(novel_images, novel_labels, transform=transform)
            novel_loader = DataLoader(novel_dataset, batch_size=1, shuffle=True, num_workers=0)

            # only leather class
            buffer_fine_tune_model = TrainerDeepSVDD(args, (novel_loader, novel_loader), device=device)
            state_dict = torch.load(buffer_fine_tune_model.trained_weights_path)
            buffer_fine_tune_model.net.load_state_dict(state_dict['net_dict'])
            buffer_fine_tune_model.c = torch.Tensor(state_dict['center']).to(buffer_fine_tune_model.device)
            
            # fine-tuning
            buffer_fine_tune_model.buffer_fine_tune()

            """ Test (Dataset: Known + Novel) """
            # Re-evaluate on the test set after fine-tuning
            fine_tune_results_dir = os.path.join(args.output_path, "IND_buffer", f"{args.testdataset_version}_fine_tune", instance)
            os.makedirs(fine_tune_results_dir, exist_ok=True)

            # Load fine-tuned model weights
            fine_tune_model = TrainerDeepSVDD(args, data_NI, device=device) 
            state_dict = torch.load(fine_tune_model.buffer_fine_tune_weights_path)
            fine_tune_model.net.load_state_dict(state_dict['net_dict'])
            fine_tune_model.c = torch.Tensor(state_dict['center']).to(fine_tune_model.device)
            
            indices, labels, scores = eval(fine_tune_model.net, fine_tune_model.c, data_NI[1], device)
            
            normal_count = 0
            anomaly_count = 0
            
            for label in labels:
                if label == 0:
                    normal_count += 1
                elif label == 1:
                    anomaly_count += 1
            
            # 각 instance에 대한 레이블의 개수 출력
            print(f"Instance: {instance}")
            print(f"  Number of normal (label 0) samples: {normal_count}")
            print(f"  Number of anomaly (label 1) samples: {anomaly_count}")

            """ Visualization """
            # Seperate normal and abnormal score
            normal_scores = [score for label, score in zip(labels, scores) if label == 0] # normal
            abnormal_scores = [score for label, score in zip(labels, scores) if label == 1] # abnormal
            normal_scores_np = np.array(normal_scores)
            mean_normal_scores_np = np.mean(normal_scores_np)
            thresholds[args.dataset_name] = mean_normal_scores_np

            # Score Distribution
            visualization.distribution_normal(normal_scores, fine_tune_results_dir)
            visualization.distribution_abnormal(abnormal_scores, fine_tune_results_dir)
            visualization.distribution_comparison(normal_scores, abnormal_scores, fine_tune_results_dir)

            # AUROC, Confusion Matrix
            visualization.auroc_confusion_matrix(args, labels, scores, thresholds, fine_tune_results_dir)

            # Top Normal(5) & Abnormal(5) 
            visualization.top5_down5_visualization(args, indices, labels, scores, data_NI, fine_tune_results_dir)

            visualization.visualize_feature_embeddings(fine_tune_model.net, data_NI[1], fine_tune_results_dir, device)
            visualization.plot_roc_curve(labels, scores, fine_tune_results_dir)
            visualization.plot_feature_distribution(fine_tune_model.net, data_NI[1], fine_tune_results_dir, device)

     