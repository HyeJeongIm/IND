import numpy as np
import argparse 
import torch

from train import TrainerDeepSVDD
# from preprocess import get_mnist
from dataloader import get_dataloaders
from dataloader_IND import get_IND_loaders, load_min_max, CustomDatasetLoader, get_only_IND_testloaders
import os
import visualization
from test import eval, eval_single_image, eval
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
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # OOD classes
    class_names = ['leather', 'metal_nut', 'wood']
    normal_classes = ['good']  # 초기 정상 데이터셋
    
    instance_folders = ['glue', 'color', 'cut', 'fold', 'poke']
    for instance in instance_folders:
        args.dataset_name = 'leather'
        print(f"Processing {instance} dataset...")
        new_instance = [instance]
        normal_classes.append(instance)
        
        data_only_NI = get_only_IND_testloaders(args, new_instance) 
        _, data_only_NI_loader = data_only_NI
        data_NI = get_IND_loaders(args, normal_classes)  # 업데이트된 normal_classes 전달

        novel_buffer = []
        results = []
        
        for idx, (image, label) in enumerate(data_only_NI[1]):
            image = image.to(device)
            scores = {}
            for class_name in class_names:
                """ Create Model & Load weight """
                args.dataset_name = class_name
                model = TrainerDeepSVDD(args, data_NI, device=device) 
                state_dict = torch.load(model.trained_weights_path)
                # print(model.trained_weights_path)
                model.net.load_state_dict(state_dict['net_dict'])
                model.c = torch.Tensor(state_dict['center']).to(model.device)

                score = eval_single_image(model.net, model.c, image, device)
                scores[class_name] = score[0]   
  
            # 각 클래스에 대한 점수의 최솟값을 선택하여 Novel 또는 Known으로 분류
            min_class_score = min(scores.values())
            
            """ 1) Known / Novel """
            if min_class_score > 102.45:  # 임의의 max 값 선택
                novel_buffer.append((data_only_NI_loader.dataset.img_paths[idx], label))  # 이미지 경로와 라벨을 버퍼에 추가
                novelty_status = "Novel"
            else:
                novelty_status = "Known"

            results.append((idx, label[0].item(), scores, novelty_status))
            # print(f"Image {idx + 1} - Label: {label[0].item()}, Scores: {scores}, Status: {novelty_status}")
        
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
        
        """ 2) OOD update """
        # Fine-tune if "Novel"
        if len(novel_buffer) > 0:
            print(f"Fine-tuning with {len(novel_buffer)} Novel images from {instance}")

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
            
            buffer_fine_tune_model.buffer_fine_tune()

            """ Test """
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

            # Score Distribution
            visualization.distribution_normal(normal_scores, fine_tune_results_dir)
            visualization.distribution_abnormal(abnormal_scores, fine_tune_results_dir)
            visualization.distribution_comparison(normal_scores, abnormal_scores, fine_tune_results_dir)

            # AUROC, Confusion Matrix
            visualization.auroc_confusion_matrix(args, labels, scores, fine_tune_results_dir)

            # Top Normal(5) & Abnormal(5) 
            visualization.top5_down5_visualization(args, indices, labels, scores, data_NI, fine_tune_results_dir)

            visualization.visualize_feature_embeddings(fine_tune_model.net, data_NI[1], fine_tune_results_dir, device)
            visualization.plot_roc_curve(labels, scores, fine_tune_results_dir)
            visualization.plot_feature_distribution(fine_tune_model.net, data_NI[1], fine_tune_results_dir, device)

        # ipdb.set_trace()


    # #######################################################

    # # Check save path
    # data_dir = os.path.join(args.dataset_path, args.dataset_name)
    # result_dir = os.path.join(args.output_path, args.dataset_name)
    # print(f"Using dataset: {data_dir}")
    # print(f"Results will be saved in: {result_dir}")
 
    # """ Load the IND test data """
    # data = get_IND_loaders(args, args.testdataset_version)
    # _, test_loader = data

    # """ Model """
    # # Class names
    # class_names = ['leather', 'metal_nut', 'wood']
    # results = []
    # novel_buffer = []

    # """ Test """
    # for idx, (image, label) in enumerate(test_loader):
    #     image = image.to(device)
    #     scores = {}

    #     for class_name in class_names:
            
    #         """ Create Model & Load weight """
    #         args.dataset_name = class_name
    #         model = TrainerDeepSVDD(args, data, device=device) 
    #         state_dict = torch.load(model.trained_weights_path)
    #         # print(model.trained_weights_path)
    #         model.net.load_state_dict(state_dict['net_dict'])
    #         model.c = torch.Tensor(state_dict['center']).to(model.device)

    #         score = eval_single_image(model.net, model.c, image, device)
    #         scores[class_name] = score[0]   
    #         # print(class_name, idx, label[0], score)
        
    #     # print(scores)
    #     # 각 클래스에 대한 점수의 최솟값을 선택하여 Novel 또는 Known으로 분류
    #     min_class_score = min(scores.values())
        
    #     """ 1) Known / Novel """
    #     if min_class_score > 102.45:  # 임의의 max 값 선택
    #         novel_buffer.append((image.cpu(), label))
    #         novelty_status = "Novel"
    #     else:
    #         novelty_status = "Known"

    #     results.append((idx, label[0].item(), scores, novelty_status))
    #     print(f"Image {idx + 1} - Label: {label[0].item()}, Scores: {scores}, Status: {novelty_status}")

    # # Save initial results to file
    # results_dir = os.path.join(args.output_path, "IND_buffer", args.testdataset_version)
    # os.makedirs(results_dir, exist_ok=True)  
    # results_path = os.path.join(results_dir, 'test_results.txt')
    # with open(results_path, 'w') as f:
    #     for idx, label, scores, novelty_status in results:
    #         min_class_score = min(scores.values())  
    #         f.write(f"Image {idx + 1} - Label: {label}, Scores: {scores}, Status: {novelty_status}, Min Score: {min_class_score:.2f}\n")

    # print(f"Initial results saved to {results_path}")
    # # Save novel images
    # visualization.visualize_and_save_novel_images(novel_buffer, results_dir)

    # """ 2) OOD update """
    # # Fine-tune if "Novel"
    # if len(novel_buffer) > 0:
    #     print(f"Fine-tuning with {len(novel_buffer)} Novel images")

    #     # Load the min_max values for normalization
    #     min_max = load_min_max(args.dataset_name, args.dataset_path)

    #     # Define the transformations
    #     transform = transforms.Compose([
    #         transforms.Resize((128, 128)),
    #         transforms.ToTensor(),
    #         transforms.Lambda(lambda x: global_contrast_normalization(x)),
    #         transforms.Normalize([min_max[0]], [min_max[1] - min_max[0]])
    #     ])

    #     # Create a new dataset and dataloader from the novel_buffer
    #     novel_images, novel_labels = zip(*novel_buffer)
    #     novel_dataset = CustomDatasetLoader(novel_images, novel_labels, transform=transform)
    #     novel_loader = DataLoader(novel_dataset, batch_size=1, shuffle=True, num_workers=0)

    #     buffer_fine_tune_model = TrainerDeepSVDD(args, data, device=device)
        
    #     # Load the pre-trained weights
    #     state_dict = torch.load(buffer_fine_tune_model.trained_weights_path)
    #     buffer_fine_tune_model.net.load_state_dict(state_dict['net_dict'])
    #     buffer_fine_tune_model.c = torch.Tensor(state_dict['center']).to(buffer_fine_tune_model.device)
        
    #     # Fine-tuning
    #     buffer_fine_tune_model.buffer_fine_tune()

    #     """ Test """
    #     # Re-evaluate on the test set after fine-tuning
    #     fine_tune_results_dir = os.path.join(args.output_path, "IND_buffer", f"{args.testdataset_version}_fine_tune")
    #     os.makedirs(fine_tune_results_dir, exist_ok=True)

    #     # Fine-tuned 모델로 테스트 수행 및 결과 저장
    #     print("Re-evaluating test set with the fine-tuned model...")
    #     test_results = []

    #     for idx, (image, label) in enumerate(test_loader):
    #         image = image.to(device)
    #         scores = {}

    #         for class_name in class_names:
    #             args.dataset_name = class_name
    #             model_ = TrainerDeepSVDD(args, data, device=device)
    #             fine_tune_model = TrainerDeepSVDD(args, data, device=device) 

    #             if args.dataset_name == 'leather':
    #                 # Load fine-tuned model weights
    #                 state_dict = torch.load(fine_tune_model.buffer_fine_tune_weights_path)
    #                 fine_tune_model.net.load_state_dict(state_dict['net_dict'])
    #                 fine_tune_model.c = torch.Tensor(state_dict['center']).to(fine_tune_model.device)
    #                 score = eval_single_image(fine_tune_model.net, fine_tune_model.c, image, device)
    #             else:
    #                 state_dict = torch.load(model_.trained_weights_path)
    #                 model_.net.load_state_dict(state_dict['net_dict'])
    #                 model_.c = torch.Tensor(state_dict['center']).to(model_.device)
    #                 score = eval_single_image(model_.net, model_.c, image, device)
    #             scores[class_name] = score[0]

    #         # Calculate the minimum score across all classes
    #         min_class_score = min(scores.values())
            
    #         # Determine if the image is Novel or Known based on the minimum class score
    #         novelty_status = "Novel" if min_class_score > 102.45 else "Known"

    #         test_results.append((idx, label[0].item(), scores, novelty_status))
    #         print(f"Fine-tuned - Image {idx + 1} - Label: {label[0].item()}, Scores: {scores}, Status: {novelty_status}")

    #     # Save fine-tuning results to file
    #     fine_tune_results_path = os.path.join(fine_tune_results_dir, 'fine_tuned_test_results.txt')
    #     with open(fine_tune_results_path, 'w') as f:
    #         for idx, label, scores, novelty_status in test_results:
    #             min_class_score = min(scores.values())  # min_class_score를 계산
    #             f.write(f"Fine-tuned - Image {idx + 1} - Label: {label}, Scores: {scores}, Status: {novelty_status}, Min Score: {min_class_score:.2f}\n")

    #     print(f"Fine-tuning results saved to {fine_tune_results_path}")

    # # OOD update 후 test
    # args.testdataset_version = 'v5'
    # data = get_IND_testloaders(args, args.testdataset_version)

    # indices, labels, scores = eval(fine_tune_model.net, fine_tune_model.c, data[1], device)
    # """ Visualization """
    # # Seperate normal and abnormal score
    # normal_scores = [score for label, score in zip(labels, scores) if label == 0] # normal
    # abnormal_scores = [score for label, score in zip(labels, scores) if label == 1] # abnormal

    # # Score Distribution
    # visualization.distribution_normal(normal_scores, fine_tune_results_dir)
    # visualization.distribution_abnormal(abnormal_scores, fine_tune_results_dir)
    # visualization.distribution_comparison(normal_scores, abnormal_scores, fine_tune_results_dir)

    # # AUROC, Confusion Matrix
    # visualization.auroc_confusion_matrix(args, labels, scores, fine_tune_results_dir)

    # # Top Normal(5) & Abnormal(5) 
    # visualization.top5_down5_visualization(args, indices, labels, scores, data, fine_tune_results_dir)
    
    # # Treshold에 따른 Misclassified Images (FP & FN)
    # def calculate_threshold(normal_scores):
    #     # 정상 샘플의 평균 점수를 threshold 값으로 사용
    #     threshold = np.mean(normal_scores)
    #     return threshold

    # threshold = calculate_threshold(normal_scores)

    # predictions = [1 if score >= threshold else 0 for score in scores]
    # visualization.visualize_misclassified(args, indices, labels, predictions, scores, data, fine_tune_results_dir)
    
    # # Usage in IND.py
    # visualization.visualize_feature_embeddings(fine_tune_model.net, data[1], fine_tune_results_dir, device)
    # visualization.plot_roc_curve(labels, scores, fine_tune_results_dir)
    # visualization.plot_feature_distribution(fine_tune_model.net, data[1], fine_tune_results_dir, device)


