import os
import torch
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Circle
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, confusion_matrix

def auroc_confusion_matrix1(args, labels, scores, result_dir):

    roc_auc = roc_auc_score(labels, scores) * 100

    # confusion matrix
    normal_scores = [score for label, score in zip(labels, scores) if label == 0]
    normal_max_dist = max(normal_scores)
    novel_scores = [score for label, score in zip(labels, scores) if label == 1]
    novel_max_dist = min(novel_scores)
    
    threshold = sum(normal_scores) / len(normal_scores)
    predictions = [1 if score >= threshold else 0 for score in scores]
    
    cf_matrix = confusion_matrix(labels, predictions)
    
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    cf_labels = ["{0}\n{1}\n({2})".format(v1, v2, v3) for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    cf_labels = np.asarray(cf_labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=cf_labels, fmt='', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Novel'], yticklabels=['Actual Normal', 'Actual Novel'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Class {args.normal_class}')
    plt.savefig(f'{result_dir}/02. Confusion_Matrix.png')
    plt.close()

    normal_scores_np = np.array(normal_scores)
    novel_scores_np = np.array(novel_scores)

    result_file_path = os.path.join(result_dir, '01. auc_scores.txt')    
    with open(result_file_path, 'w') as file:
        file.write(f' AUROC: {roc_auc:.2f}%\n')
        file.write(f'Label counts: {Counter(labels)}\n')
        file.write(f'Prediction counts: {Counter(predictions)}\n')
        file.write(f'Normal Scores - Min: {np.min(normal_scores_np):.2f}, Max: {np.max(normal_scores_np):.2f}, Mean: {np.mean(normal_scores_np):.2f}\n')
        file.write(f'Novel Scores - Min: {np.min(novel_scores_np):.2f}, Max: {np.max(novel_scores_np):.2f}, Mean: {np.mean(novel_scores_np):.2f}\n')

def auroc_confusion_matrix(args, labels, scores, thresholds, result_dir):

    roc_auc = roc_auc_score(labels, scores) * 100

    # confusion matrix
    normal_scores = [score for label, score in zip(labels, scores) if label == 0]
    normal_max_dist = max(normal_scores)
    novel_scores = [score for label, score in zip(labels, scores) if label == 1]
    novel_max_dist = min(novel_scores)
    
    threshold = sum(normal_scores) / len(normal_scores)
    predictions = [1 if score >= threshold else 0 for score in scores]
    
    cf_matrix = confusion_matrix(labels, predictions)
    
    group_names = ["TN", "FP", "FN", "TP"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    cf_labels = ["{0}\n{1}\n({2})".format(v1, v2, v3) for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    cf_labels = np.asarray(cf_labels).reshape(2,2)
    
    sns.heatmap(cf_matrix, annot=cf_labels, fmt='', cmap='Blues', xticklabels=['Predicted Normal', 'Predicted Novel'], yticklabels=['Actual Normal', 'Actual Novel'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Class {args.normal_class}')
    plt.savefig(f'{result_dir}/02. Confusion_Matrix.png')
    plt.close()

    normal_scores_np = np.array(normal_scores)
    novel_scores_np = np.array(novel_scores)

    result_file_path = os.path.join(result_dir, '01. auc_scores.txt')    
    with open(result_file_path, 'w') as file:
        file.write(f' AUROC: {roc_auc:.2f}%\n')
        file.write(f'Label counts: {Counter(labels)}\n')
        file.write(f'Prediction counts: {Counter(predictions)}\n')
        file.write(f'Normal Scores - Min: {np.min(normal_scores_np):.2f}, Max: {np.max(normal_scores_np):.2f}, Mean: {np.mean(normal_scores_np):.2f}\n')
        file.write(f'Novel Scores - Min: {np.min(novel_scores_np):.2f}, Max: {np.max(novel_scores_np):.2f}, Mean: {np.mean(novel_scores_np):.2f}\n')
        file.write(f'Threshold: {thresholds}\n')

def distribution_normal(normal_scores, result_dir):
    """
    Normal 점수 분포 시각화 및 저장

    Args:
        normal_scores (list or np.array): Normal 점수 목록
        result_dir (str): 결과를 저장할 디렉토리 경로
    """
    plt.figure(figsize=(12, 8))
    
    # KDE Plot
    sns.kdeplot(normal_scores, fill=True, color='blue', label='Normal')
    
    # 히스토그램
    plt.hist(normal_scores, bins=30, alpha=0.5, color='blue', edgecolor='black', density=True)
    
    plt.title('Distribution of Normal Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_dir}/03_01. normal_score_distribution.png')
    plt.close()

def distribution_novel(novel_scores, result_dir):
    """
    Novel 점수 분포 시각화 및 저장

    Args:
        novel_scores (list or np.array): Novel 점수 목록
        result_dir (str): 결과를 저장할 디렉토리 경로
    """
    plt.figure(figsize=(12, 8))
    
    # KDE Plot
    sns.kdeplot(novel_scores, fill=True, color='red', label='Novel')
    
    # 히스토그램
    plt.hist(novel_scores, bins=30, alpha=0.5, color='red', edgecolor='black', density=True)
    
    plt.title('Distribution of Novel Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_dir}/03_02.novel_score_distribution.png')
    plt.close()

def distribution_comparison(normal_scores, novel_scores, result_dir):
    """
    Normal 및 Novel 점수 분포 비교 시각화 및 저장

    Args:
        normal_scores (list or np.array): Normal 점수 목록
        novel_scores (list or np.array): Novel 점수 목록
        result_dir (str): 결과를 저장할 디렉토리 경로
    """
    plt.figure(figsize=(12, 8))
    
    # KDE Plot
    sns.kdeplot(normal_scores, fill=True, color='blue', label='Normal')
    sns.kdeplot(novel_scores, fill=True, color='red', label='Novel')
    
    plt.title('Comparison of Normal and Novel Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_dir}/03_03. score_distribution_comparison.png')
    plt.close()

# 가장 이상적인 normal, novel class 5개 시각화 
def top5_down5_visualization(args, indices, labels, scores, data, result_dir):
    normal_indices_scores = [(idx, score) for idx, score, label in zip(indices, scores, labels) if label == 0]
    novel_indices_scores = [(idx, score) for idx, score, label in zip(indices, scores, labels) if label == 1]

    normal_indices_scores.sort(key=lambda x: x[1])  
    novel_indices_scores.sort(key=lambda x: x[1], reverse=True)  
    top_normal_images_scores = normal_indices_scores[:5]
    top_novel_images_scores = novel_indices_scores[:5]

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for i, (idx, score) in enumerate(top_normal_images_scores):
        img = data[1].dataset[idx][0]
        img = img.permute(1, 2, 0).cpu().numpy()  # Adjust for matplotlib
        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Normal\nScore: {score:.2f}')
        axs[0, i].axis('off')

    for i, (idx, score) in enumerate(top_novel_images_scores):
        img = data[1].dataset[idx][0]
        img = img.permute(1, 2, 0).cpu().numpy()  # Adjust for matplotlib
        axs[1, i].imshow(img)
        axs[1, i].set_title(f'Novel\nScore: {score:.2f}')
        axs[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{result_dir}/04. visualization.png')
    plt.close(fig)

    print(f'Finished')

def visualize_misclassified(args, indices, labels, predictions, scores, data, result_dir):
    # 잘못 분류된 사례를 저장할 리스트 초기화
    false_positives = []  # 0인데 1로 분류한 경우
    false_negatives = []  # 1인데 0으로 분류한 경우

    for idx, label, prediction, score in zip(indices, labels, predictions, scores):
        if label == 0 and prediction == 1:
            false_positives.append((idx, score))
        elif label == 1 and prediction == 0:
            false_negatives.append((idx, score))

    # 상위 5개만 선택, 부족하면 빈 칸으로 채움
    false_positives = false_positives[:5] + [None] * (5 - len(false_positives))
    false_negatives = false_negatives[:5] + [None] * (5 - len(false_negatives))

    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    # 0인데 1로 잘못 분류된 경우 시각화
    for i, item in enumerate(false_positives):
        if item is not None:
            idx, score = item
            img = data[1].dataset[idx][0]
            img = img.permute(1, 2, 0).cpu().numpy()  # Adjust for matplotlib
            axs[0, i].imshow(img)
            axs[0, i].set_title(f'FP (0 -> 1)\nScore: {score:.2f}')
            axs[0, i].axis('off')
        else:
            axs[0, i].axis('off')  # 빈 공간

    # 1인데 0으로 잘못 분류된 경우 시각화
    for i, item in enumerate(false_negatives):
        if item is not None:
            idx, score = item
            img = data[1].dataset[idx][0]
            img = img.permute(1, 2, 0).cpu().numpy()  # Adjust for matplotlib
            axs[1, i].imshow(img)
            axs[1, i].set_title(f'FN (1 -> 0)\nScore: {score:.2f}')
            axs[1, i].axis('off')
        else:
            axs[1, i].axis('off')  # 빈 공간

    plt.tight_layout()
    plt.savefig(f'{result_dir}/05. misclassified_visualization.png')
    plt.close(fig)
    print(f'Misclassified examples visualization saved at {result_dir}/misclassified_visualization.png')

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_embeddings(net, dataloader, save_dir, device):
    net.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, label in dataloader:
            x = x.float().to(device)
            z = net(x).cpu().numpy()
            features.append(z)
            labels.append(label.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Apply t-SNE or UMAP
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 8))
    plt.scatter(features_2d[labels==0, 0], features_2d[labels==0, 1], label='Normal', alpha=0.5)
    plt.scatter(features_2d[labels==1, 0], features_2d[labels==1, 1], label='Anomalous', alpha=0.5, color='r')
    plt.legend()
    plt.title('t-SNE of Latent Features')
    plt.savefig(os.path.join(save_dir, '06. tsne_feature_embeddings.png'))
    plt.close()

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(labels, scores, save_dir, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, '07. roc_curve.png'))
    plt.close()

def plot_feature_distribution(net, dataloader, save_dir, device):
    net.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, label in dataloader:
            x = x.float().to(device)
            z = net(x).cpu().numpy()
            features.append(z)
            labels.append(label.numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(10, 6))
    plt.hist(features[labels==0].flatten(), bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist(features[labels==1].flatten(), bins=50, alpha=0.5, label='Anomalous', color='red')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Feature Distribution')
    plt.savefig(os.path.join(save_dir, '08. feature_distribution.png'))
    plt.close()

import matplotlib.pyplot as plt
from PIL import Image

def visualize_and_save_novel_images(novel_buffer, results_dir):
    dir = os.path.join(results_dir, "Novel_img")

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    for i, (img_path, label) in enumerate(novel_buffer):
        image = Image.open(img_path).convert('RGB')
        plt.figure()
        plt.imshow(image)

        plt.title(f"Novel Image {i+1} - Label: {label.item()}")
        plt.axis('off')

        # Save the image
        image_path = os.path.join(dir, f"novel_image_{i+1}_label_{label.item()}.png")
        plt.savefig(image_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved {image_path}")

def visualize_and_save_known_images(known_buffer, results_dir):
    dir = os.path.join(results_dir, "Known_img")

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    for i, (img_path, label) in enumerate(known_buffer):
        image = Image.open(img_path).convert('RGB')
        plt.figure()
        plt.imshow(image)

        plt.title(f"Kovel Image {i+1} - Label: {label.item()}")
        plt.axis('off')

        # Save the image
        image_path = os.path.join(dir, f"known_image_{i+1}_label_{label.item()}.png")
        plt.savefig(image_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved {image_path}")


def visualize_latent_space_train(net, center, train_loader, device, save_dir):
    net.eval()
    radii = []
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.float().to(device)
            z = net(x)  # train dataset을 latent space로 변환
            distances = torch.sum((z - center) ** 2, dim=1)  # 중심과의 거리 계산 (반지름 값)
            radii.extend(distances.cpu().numpy())  # 반지름 값 저장

    radii = np.array(radii)

    # 가장 작은 반지름과 가장 큰 반지름 계산
    min_radius = np.min(radii)
    max_radius = np.max(radii)

    # KDE를 사용하여 반지름의 밀도 추정
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(radii[:, None])
    log_density = kde.score_samples(radii[:, None])
    density = np.exp(log_density)
    
    # 가장 밀도가 높은 반지름 찾기
    most_common_radius = radii[np.argmax(density)]

    # 2D 좌표 생성 (임의의 원 형태로 변환)
    angles = np.linspace(0, 2 * np.pi, len(radii))
    x_coords = radii * np.cos(angles)
    y_coords = radii * np.sin(angles)

    # 시각화
    plt.figure(figsize=(8, 8), facecolor='white')  # 배경색을 흰색으로 설정

    # normal 점들 시각화
    scatter = plt.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.6, edgecolor='black', linewidth=0.5, label='Normal')

    # 중심 표시
    plt.scatter(0, 0, c='black', s=150, marker='o', label='Center', linewidth=2)

    # 가장 작은 반지름과 가장 큰 반지름을 기준으로 원 추가
    circle_min = Circle((0, 0), min_radius, color='green', fill=False, linestyle='--', linewidth=2, label=f'Min Radius ({min_radius:.2f})')
    circle_max = Circle((0, 0), max_radius, color='red', fill=False, linestyle='--', linewidth=2, label=f'Max Radius ({max_radius:.2f})')
    circle_common = Circle((0, 0), most_common_radius, color='orange', fill=False, linestyle='--', linewidth=2, label=f'Most Common Radius ({most_common_radius:.2f})')

    plt.gca().add_patch(circle_min)
    plt.gca().add_patch(circle_max)
    plt.gca().add_patch(circle_common)

    # 축과 그래프의 스타일 설정
    plt.title('Latent Space Visualization of Train Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    # 이미지 저장 및 출력
    plt.savefig(os.path.join(save_dir, '09. latent space train dataset.png'), bbox_inches='tight', dpi=300)  # 해상도를 높여서 저장
    plt.show()


def visualize_latent_space_test(net, center, test_loader, device, save_dir):
    net.eval()
    radii = []
    labels_list = []
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.float().to(device)
            z = net(x)  # test dataset을 latent space로 변환
            distances = torch.sum((z - center) ** 2, dim=1)  # 중심과의 거리 계산 (반지름 값)
            radii.extend(distances.cpu().numpy())  # 반지름 값 저장
            labels_list.extend(labels.numpy())  # 라벨 저장

    radii = np.array(radii)
    labels_list = np.array(labels_list)

    # normal과 novel 점 분리
    normal_mask = (labels_list == 0)
    novel_mask = (labels_list == 1)

    normal_radii = radii[normal_mask]
    novel_radii = radii[novel_mask]

    # KDE를 사용하여 밀도가 높은 반지름 값 찾기
    kde = gaussian_kde(normal_radii)
    x_range = np.linspace(np.min(normal_radii), np.max(normal_radii), 1000)
    kde_values = kde(x_range)
    most_common_radius = x_range[np.argmax(kde_values)]

    # 2D 좌표 생성 (임의의 원 형태로 변환)
    angles_normal = np.linspace(0, 2 * np.pi, len(normal_radii))
    x_coords_normal = normal_radii * np.cos(angles_normal)
    y_coords_normal = normal_radii * np.sin(angles_normal)

    angles_novel = np.linspace(0, 2 * np.pi, len(novel_radii))
    x_coords_novel = novel_radii * np.cos(angles_novel)
    y_coords_novel = novel_radii * np.sin(angles_novel)

    # normal의 최소 반지름과 최대 반지름 계산
    min_radius = np.min(normal_radii)
    max_radius = np.max(normal_radii)

    # 시각화
    plt.figure(figsize=(8, 8), facecolor='white')  # 배경색을 흰색으로 설정

    # normal과 novel 점들 시각화
    scatter_normal = plt.scatter(x_coords_normal, y_coords_normal, c='blue', s=50, alpha=0.6, edgecolor='black', linewidth=0.5, label='Normal')
    scatter_novel = plt.scatter(x_coords_novel, y_coords_novel, c='red', s=50, alpha=0.6, edgecolor='black', linewidth=0.5, label='Novel')

    # 중심 표시
    plt.scatter(0, 0, c='black', s=100, marker='o', label='Center', linewidth=2)  # 중심 원은 조금 작게 설정

    # 가장 작은 반지름과 가장 큰 반지름을 기준으로 원 추가
    circle_min = Circle((0, 0), min_radius, color='green', fill=False, linestyle='--', linewidth=2, label=f'Min Normal Radius ({min_radius:.2f})')
    circle_max = Circle((0, 0), max_radius, color='red', fill=False, linestyle='--', linewidth=2, label=f'Max Normal Radius ({max_radius:.2f})')
    circle_common = Circle((0, 0), most_common_radius, color='orange', fill=False, linestyle='--', linewidth=2, label=f'Most Common Radius ({most_common_radius:.2f})')

    plt.gca().add_patch(circle_min)
    plt.gca().add_patch(circle_max)
    plt.gca().add_patch(circle_common)

    # 축과 그래프의 스타일 설정
    plt.title('Latent Space Visualization of Test Dataset', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X Coordinate', fontsize=14)
    plt.ylabel('Y Coordinate', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)

    # 이미지 저장 및 출력
    plt.savefig(os.path.join(save_dir, '10. latent space test dataset.png'), bbox_inches='tight', dpi=300)  # 해상도를 높여서 저장
    plt.show()