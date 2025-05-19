from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def test_model():
    # Load the model and feature extractor
    model_path = "output-models/checkpoint-9400"
    model = AutoModelForImageClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    
    # Load the test dataset
    dataset = load_dataset("datasets/erm570", split="test")
    
    # Prepare the data
    def preprocess_function(examples):
        # Process images in batches
        images = examples["image"]
        processed = feature_extractor(images, return_tensors="pt")
        return {
            "pixel_values": processed["pixel_values"].squeeze(0),  # Remove batch dimension
            "label": examples["label"]
        }
    
    # Preprocess the dataset
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset.column_names
    )

    # Convert to PyTorch tensors
    processed_dataset.set_format(type="torch")

    # Create data loader
    test_dataloader = DataLoader(processed_dataset, batch_size=8)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize lists for predictions and true labels
    all_predictions = []
    all_labels = []
    
    # Test the model
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Move batch to device
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["label"]
            
            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    # Row-wise normalization
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['Benign', 'Malignant', 'Normal'],
                                 digits=3)
    
    # Print results
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nNormalized Confusion Matrix:")
    print(conf_matrix_norm)
    print("\nClassification Report:")
    print(report)
    
    # Plot and save normalized confusion matrix
    classes = ['Benign', 'Malignant', 'Normal']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_test_vit_normalized.png')
    plt.close()
    
    # Save results to file
    with open('test_results.txt', 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nNormalized Confusion Matrix:\n")
        f.write(str(conf_matrix_norm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

if __name__ == "__main__":
    test_model() 