import os
import random
import torch
import argparse 
import numpy as np
import visualization
from train import TrainerDeepSVDD
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.neighbors import KernelDensity
from utils.utils import global_contrast_normalization
from test import eval, eval_single_image, eval, eval_single_image_IDPOC
from dataloader_IND import get_IND_loaders, load_min_max, CustomDatasetLoader, get_only_IND_testloaders

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
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    """
        - 초기 class 및 normal class 정의 
        - 초기 threshold 설정 
            - normal data의 most, max의 평균값
    """
    normal_classes = ['good']  
    class_names = ['leather', 'metal_nut', 'wood']

    thresholds = {
        'leather': 22.82,
        'metal_nut': 10.88,
        'wood': 32.22
    }
    
    # new instance class 정의 
    instance_folders = ['glue', 'color', 'cut', 'fold', 'poke']
    for instance in instance_folders:
        args.dataset_name = 'leather' # 수정 필요 
        print(f"Processing {instance} dataset...")
        
        novel_buffer = []
        known_buffer = [] 
        results = []

        new_instance = [instance]
        normal_classes.append(instance)

        """ 
            Load Data 
            - get_IND_loaders: normal + novel(new instance) dataset
            - get_only_IND_testloaders: only novel(new instance) dataset
        """
        data_NI = get_IND_loaders(args, normal_classes)  
        data_only_NI = get_only_IND_testloaders(args, new_instance) 
        _, data_only_NI_loader = data_only_NI
        
        for idx, (image, label) in enumerate(data_only_NI[1]):
            image = image.to(device)
            scores = {}
            is_novel = True  # 이미지가 모든 모델에서 novel로 분류되는지 여부 확인 가능 

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
        
        """
            Save result
            - new instance에 대해서만 Known or Novel 구분한 결과
        """
        results_dir = os.path.join(args.output_path, "IND_buffer", args.testdataset_version, instance)
        os.makedirs(results_dir, exist_ok=True)  
   
        results_path = os.path.join(results_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            for idx, label, scores, novelty_status in results:
                min_class_score = min(scores.values())  
                f.write(f"Image {idx + 1} - Label: {label}, Scores: {scores}, Status: {novelty_status}, Min Score: {min_class_score:.2f}\n")
            
            # known_buffer와 novel_buffer에 담긴 이미지 개수 추가
            f.write(f"\nTotal Known Images: {len(known_buffer)}\n")
            f.write(f"Total Novel Images: {len(novel_buffer)}\n")

        # Save known/novel images
        visualization.visualize_and_save_known_images(known_buffer, results_dir)
        visualization.visualize_and_save_novel_images(novel_buffer, results_dir)

        print(f"Initial results saved to {results_path}")


        """ 
            2) OOD update 
                - novel_buffer에 novel image가 있는 경우
                  1) update 진행 (fine tuning)
                  2) threshold update
        """
        # Fine-tune if "Novel"
        if len(novel_buffer) > 0:
            print(f"Fine-tuning with {len(novel_buffer)} Novel images from {instance}")
            args.dataset_name = 'leather' # 수정 필요 
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

            """ 
                Test 
                - normal + new instance를 normal로 fine tuning 했을 때 다양한 결과 
            """
            # fine_tune_results_dir = os.path.join(args.output_path, "IND_buffer", f"{args.testdataset_version}_fine_tune", instance)
            fine_tune_results_dir = os.path.join(args.output_path, "IND_buffer", f"{args.testdataset_version}_fine_tune_th_kde", instance)

            os.makedirs(fine_tune_results_dir, exist_ok=True)

            # Load fine-tuned model weights
            fine_tune_model = TrainerDeepSVDD(args, data_NI, device=device) 
            state_dict = torch.load(fine_tune_model.buffer_fine_tune_weights_path)
            fine_tune_model.net.load_state_dict(state_dict['net_dict'])
            fine_tune_model.c = torch.Tensor(state_dict['center']).to(fine_tune_model.device)
            
            indices, labels, scores = eval(fine_tune_model.net, fine_tune_model.c, data_NI[1], device)

            """ Calculate threshold """
            # Seperate normal and abnormal score
            normal_scores = [score for label, score in zip(labels, scores) if label == 0] # normal
            abnormal_scores = [score for label, score in zip(labels, scores) if label == 1] # abnormal
            normal_scores_np = np.array(normal_scores)
            
            # max R 
            max_radius = np.max(normal_scores_np)
            
            # KDE를 사용하여 반지름의 밀도 추정
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(normal_scores_np[:, None])
            log_density = kde.score_samples(normal_scores_np[:, None])
            density = np.exp(log_density)
            # 가장 밀도가 높은 반지름 찾기
            most_common_radius = normal_scores_np[np.argmax(density)]

            """ threshold 계산 및 update """
            # threshold_update_value = (max_radius + most_common_radius) / 2
            # thresholds[args.dataset_name] = threshold_update_value
            thresholds[args.dataset_name] = most_common_radius

            """ Visualization """
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
            visualization.plot_feature_distribution(fine_tune_model.net, data_NI[1], fine_tune_results_dir, device)
            
            # min, max, most common radius에 따른 circle
            visualization.visualize_latent_space_train(fine_tune_model.net, fine_tune_model.c, data_NI[0], device, fine_tune_results_dir)
            visualization.visualize_latent_space_test(fine_tune_model.net, fine_tune_model.c, data_NI[1], device, fine_tune_results_dir)