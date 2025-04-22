import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_training_loss(json_path):
    """
    Plot training loss over epochs from trainer_state.json
    
    Args:
        json_path (str): Path to the trainer_state.json file
    """
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract epochs and losses
    epochs = []
    losses = []
    
    for entry in data['log_history']:
        if 'loss' in entry:
            epochs.append(entry['epoch'])
            losses.append(entry['loss'])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    
    # Customize the plot
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.savefig('training_loss.png')
    plt.close()

def plot_confusion_matrix():
    """
    Plot the confusion matrix with normalized values
    """
    # Define the confusion matrix
    cm = np.array([
        [80, 5, 2],
        [5, 37, 0],
        [4, 2, 21]
    ])
    
    # Calculate row-wise normalization (percentage of each true class)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create labels (assuming these are the classes)
    classes = ['Benign', 'Malignant', 'Normal']
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    # Customize the plot
    plt.title('Confusion Matrix')

    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    #plot_training_loss('output-models/checkpoint-9400/trainer_state.json')
    plot_confusion_matrix()
