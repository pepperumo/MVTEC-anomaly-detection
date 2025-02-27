import os
import torch
import dill
import pickle
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

def decision_function(segm_map):
    """
    Calculate anomaly score from segmentation map using mean of top 10 values.
    """
    mean_top_10_values = []
    for map in segm_map:
        flattened_tensor = map.reshape(-1)
        sorted_tensor, _ = torch.sort(flattened_tensor, descending=True)
        mean_top_10_value = sorted_tensor[:10].mean()
        mean_top_10_values.append(mean_top_10_value)

    return torch.stack(mean_top_10_values)

def run_inference_autoencoder(image_path, model, backbone, threshold):
    """
    Run inference on a single image using Autoencoder.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    test_image = transform(image).cuda().unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        features = backbone(test_image)
        recon = model(features)

        # Compute segmentation map and anomaly score
        segm_map = ((features - recon) ** 2).mean(axis=(1))
        y_score = decision_function(segm_map=segm_map)
        is_anomaly = (y_score >= threshold).cpu().numpy().item()

        # Create heatmap
        heat_map = cv2.resize(segm_map.squeeze().cpu().numpy(), (224, 224))

    return {
        'original_image': test_image.squeeze().permute(1, 2, 0).cpu().numpy(),
        'heat_map': heat_map,
        'anomaly_score': y_score.item(),
        'threshold': threshold,
        'is_anomaly': is_anomaly,
        'classification': 'NOK' if is_anomaly else 'OK'
    }

def run_inference_knn(image_path, model, memory_bank, threshold):
    """
    Run inference on a single image using KNN.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    test_image = transform(image).cuda().unsqueeze(0)

    # Move memory bank to GPU **only once**
    memory_bank = memory_bank.cuda()

    # Extract features using the backbone
    with torch.no_grad():
        features = model(test_image)

    # Compute distances (optimized)
    distances = torch.cdist(features, memory_bank, p=2.0)  # Batched distance calculation
    dist_score, _ = torch.min(distances, dim=1)  # Get the nearest neighbor
    y_score = torch.max(dist_score)  # Get the anomaly score

    is_anomaly = (y_score >= threshold).cpu().item()

    # Compute segmentation map
    segm_map = dist_score.view(1, 1, 28, 28)
    segm_map = torch.nn.functional.interpolate(segm_map, size=(224, 224), mode='bilinear').cpu().squeeze().numpy()
    
    # Convert segmentation map to heatmap
    heat_map = cv2.resize(segm_map, (224, 224))

    return {
        'original_image': test_image.squeeze().permute(1, 2, 0).cpu().numpy(),
        'heat_map': heat_map,
        'anomaly_score': y_score.item(),
        'threshold': threshold,
        'is_anomaly': is_anomaly,
        'classification': 'NOK' if is_anomaly else 'OK',
    }


def load_model_autoencoder(checkpoint_dir='models'):
    """
    Load the saved models and evaluation metrics for Autoencoder.
    """
    models_path = os.path.join(checkpoint_dir, 'models_autoencoder.dill')
    backbone_path = os.path.join(checkpoint_dir, 'backbone_autoencoder.dill')
    metrics_path = os.path.join(checkpoint_dir, 'evaluation_metrics_autoencoder.pkl')

    # Ensure files exist before loading
    if not os.path.exists(models_path):
        raise FileNotFoundError(f"Autoencoder model file not found: {models_path}")
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"Autoencoder backbone file not found: {backbone_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Autoencoder metrics file not found: {metrics_path}")

    try:
        with open(models_path, 'rb') as f:
            models = dill.load(f)

        with open(backbone_path, 'rb') as f:
            backbone = dill.load(f)

        with open(metrics_path, 'rb') as f:
            evaluation_metrics = pickle.load(f)

        return models, backbone, evaluation_metrics

    except Exception as e:
        raise RuntimeError(f"Error loading Autoencoder models: {e}")

def load_model_knn(checkpoint_dir='models'):
    """
    Load the saved models and evaluation metrics for KNN.
    """
    memory_bank_path = os.path.join(checkpoint_dir, 'memory_bank_selected.pkl')
    backbone_path = os.path.join(checkpoint_dir, 'backbone_knn.dill')  # Fixed incorrect backbone file
    metrics_path = os.path.join(checkpoint_dir, 'evaluation_metrics_knn.pkl')

    # Ensure files exist before loading
    if not os.path.exists(memory_bank_path):
        raise FileNotFoundError(f"KNN memory bank file not found: {memory_bank_path}")
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"KNN backbone file not found: {backbone_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"KNN metrics file not found: {metrics_path}")

    try:
        with open(memory_bank_path, 'rb') as f:
            memory_bank = pickle.load(f)

        with open(backbone_path, 'rb') as f:
            backbone = dill.load(f)

        with open(metrics_path, 'rb') as f:
            evaluation_metrics = pickle.load(f)

        return backbone, memory_bank, evaluation_metrics

    except Exception as e:
        raise RuntimeError(f"Error loading KNN models: {e}")
