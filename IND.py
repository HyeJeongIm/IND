import numpy as np
import argparse 
import torch

from train import TrainerDeepSVDD
# from preprocess import get_mnist
from dataloader import get_dataloaders
from dataloader_IND import get_IND_loaders
import os
import visualization
from test import eval, eval_single_image

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
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True, help='Whether to pretrain the network using an autoencoder')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of the latent variable z')
    # Dataset
    parser.add_argument('--dataset_path', type=str, default='./data/MVTecAD', help='Path to the dataset root directory')
    parser.add_argument('--dataset_name', type=str, default='leather', choices=['leather', 'metal_nut', 'wood'], help='Name of the dataset to use')
    parser.add_argument('--dataset_test_path', type=str, default='./data/IND', help='Path to the dataset root directory')
    parser.add_argument('--testdataset_version', type=str, default='v4', help='Version of the IND test dataset')

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
 
    """ Load the IND test data """
    data = get_IND_loaders(args, args.testdataset_version)
    _, test_loader = data

    """ Model """
    # Class names
    class_names = ['leather', 'metal_nut', 'wood']
    results = []

    for idx, (image, label) in enumerate(test_loader):
        image = image.to(device)
        scores = {}

        for class_name in class_names:
            
            """ Create Model & Load weight """
            args.dataset_name = class_name
            model = TrainerDeepSVDD(args, data, device=device) 
            state_dict = torch.load(model.trained_weights_path)
            # print(model.trained_weights_path)
            model.net.load_state_dict(state_dict['net_dict'])
            model.c = torch.Tensor(state_dict['center']).to(model.device)

            score = eval_single_image(model.net, model.c, image, device)
            scores[class_name] = score[0]   
            # print(class_name, idx, label[0], score)

        results.append((idx, label[0].item(), scores))
        print(f"Image {idx + 1} - Label: {label[0].item()}, Scores: {scores}")

    """ Calculate average scores for each class """
    average_scores = {class_name: 0.0 for class_name in class_names}
    num_images = len(results)

    for _, _, scores in results:
        for class_name, score in scores.items():
            average_scores[class_name] += score

    average_scores = {class_name: total / num_images for class_name, total in average_scores.items()}
    # Find the minimum average score
    min_avg_score_class = min(average_scores, key=average_scores.get)
    min_avg_score = average_scores[min_avg_score_class]

    """ Known / Novel """
    # 임의의 max 값 선택 
    # leather: 35.12, metal_nut: 10.56, wood: 12.37
    threshold = 35.12
    
    novelty_status = "Novel" if min_avg_score > threshold else "Known"

    # Save results to file
    results_dir = os.path.join(args.output_path, "IND", args.testdataset_version)
    os.makedirs(results_dir, exist_ok=True)  

    results_path = os.path.join(results_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        for idx, label, scores in results:
            f.write(f"Image {idx + 1} - Label: {label}, Scores: {scores}\n")
        
        # Write the average scores 
        f.write("\nAverage Scores across all images:\n")
        f.write(f"Label: Averages, Scores: {{")
        f.write(", ".join([f"'{class_name}': {avg_score:.4f}" for class_name, avg_score in average_scores.items()]))
        f.write("}\n")
        
        # Write the minimum average score and novelty status
        f.write(f"\nMinimum Average Score: {min_avg_score:.4f}\n")
        f.write(f"Novelty Status: {novelty_status}\n")

    print(f"Results saved to {results_path}")
    print(f"Minimum Average Score: {min_avg_score_class} with a score of {min_avg_score:.4f}")
    print(f"Novelty Status: {novelty_status}")

    # Fine-tune if "Novel"
    if novelty_status == "Novel":
        print(f"Fine-tuning for class: {min_avg_score_class}")

        # Update args for fine-tuning
        args.dataset_name = min_avg_score_class
        fine_tune_model = TrainerDeepSVDD(args, data, device=device)
        
        # Load the pre-trained weights
        state_dict = torch.load(fine_tune_model.trained_weights_path)
        fine_tune_model.net.load_state_dict(state_dict['net_dict'])
        fine_tune_model.c = torch.Tensor(state_dict['center']).to(fine_tune_model.device)
        
        # Fine-tuning
        fine_tune_model.train()

        """ Test """
        # Re-evaluate on the test set after fine-tuning
        fine_tune_results_dir = os.path.join(args.output_path, "IND", f"{args.testdataset_version}_fine_tune")
        os.makedirs(fine_tune_results_dir, exist_ok=True)

        fine_tune_results = []
        fine_tune_average_score = 0.0  

        for idx, (image, label) in enumerate(test_loader):
            image = image.to(device)
            score = eval_single_image(fine_tune_model.net, fine_tune_model.c, image, device)
            fine_tune_results.append((idx, label[0].item(), score[0]))
            fine_tune_average_score += score[0]  
            print(f"Fine-tuned - Image {idx + 1} - Label: {label[0].item()}, Score: {score[0]}")

        # Calculate the average score after fine-tuning
        fine_tune_average_score /= len(test_loader)

        # Save fine-tuning results to file
        fine_tune_results_path = os.path.join(fine_tune_results_dir, 'fine_tune_results.txt')
        with open(fine_tune_results_path, 'w') as f:
            for idx, label, score in fine_tune_results:
                f.write(f"Fine-tuned - Image {idx + 1} - Label: {label}, Score: {score:.4f}\n")

            # Write the average score after fine-tuning
            f.write(f"\nAverage Score after Fine-tuning: {fine_tune_average_score:.4f}\n")

            new_novelty_status = "Novel" if fine_tune_average_score > threshold else "Known"
            f.write(f"Novelty Status after Fine-tuning: {new_novelty_status}\n")

        print(f"Fine-tune results saved to {fine_tune_results_path}")
        print(f"Average Score after Fine-tuning: {fine_tune_average_score:.4f}")
        print(f"Novelty Status after Fine-tuning: {new_novelty_status}")
