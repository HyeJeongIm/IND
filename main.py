import numpy as np
import argparse 
import torch

from train import TrainerDeepSVDD
# from preprocess import get_mnist
from dataloader import get_dataloaders
import os
import visualization
from test import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep SVDD Training on MVTecAD Datasets")
    
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=100, help="Number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6, help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3, help='Weight decay hyperparameter for the L2 regularization during autoencoder pretraining')
    parser.add_argument('--lr_ae', type=float, default=1e-4, help='Learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50], help='Milestones at which the scheduler multiplies the learning rate by 0.1')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True, help='Whether to pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of the latent variable z')
    # Dataset
    parser.add_argument('--dataset_path', type=str, default='./data/MVTecAD', help='Path to the dataset root directory')
    parser.add_argument('--dataset_name', type=str, default='leather', choices=['leather', 'metal_nut', 'wood'], help='Name of the dataset to use')
    # Save path
    parser.add_argument('--output_path', type=str, default='./results/', help='Path to save the output models and results')
    parser.add_argument('--normal_class', type=int, default=0, help='Class to be treated as normal. The rest will be considered as anomalous.') 
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check save path
    data_dir = os.path.join(args.dataset_path, args.dataset_name)
    result_dir = os.path.join(args.output_path, args.dataset_name)
    print(f"Using dataset: {data_dir}")
    print(f"Results will be saved in: {result_dir}")
 
    """ Load data """
    data = get_dataloaders(args)
    # data = get_mnist(args)

    """ Model """
    deep_SVDD = TrainerDeepSVDD(args, data, device)

    if args.pretrain:
        deep_SVDD.pretrain()
    deep_SVDD.train()

    """ Test """
    state_dict = torch.load(deep_SVDD.trained_weights_path)
    deep_SVDD.net.load_state_dict(state_dict['net_dict'])
    deep_SVDD.c = torch.Tensor(state_dict['center']).to(deep_SVDD.device)

    indices, labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)

    """ Visualization """
    # Seperate normal and abnormal score
    normal_scores = [score for label, score in zip(labels, scores) if label == 0] # normal
    abnormal_scores = [score for label, score in zip(labels, scores) if label == 1] # abnormal

    # Score Distribution
    visualization.distribution_normal(normal_scores, result_dir)
    visualization.distribution_abnormal(abnormal_scores, result_dir)
    visualization.distribution_comparison(normal_scores, abnormal_scores, result_dir)

    # AUROC, Confusion Matrix
    visualization.auroc_confusion_matrix(args, labels, scores, result_dir)

    # Top Normal(5) & Abnormal(5) 
    visualization.top5_down5_visualization(args, indices, labels, scores, data, result_dir)
    
    # Treshold에 따른 Misclassified Images (FP & FN)
    def calculate_threshold(normal_scores):
        # 정상 샘플의 평균 점수를 threshold 값으로 사용
        threshold = np.mean(normal_scores)
        return threshold

    threshold = calculate_threshold(normal_scores)

    predictions = [1 if score >= threshold else 0 for score in scores]
    visualization.visualize_misclassified(args, indices, labels, predictions, scores, data, result_dir)
