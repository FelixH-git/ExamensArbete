from ultralytics import YOLO
import os

def test_model(model_path):
    # Load the model
    model = YOLO(model_path)
    
    # Path to the test dataset
    test_dataset_path = os.path.join('datasets', 'erm570', 'test')
    
    # Run validation on the test dataset
    results = model.val(
        data=os.path.join('datasets', 'dataset.yaml'),
        split='test',
        imgsz=640,
        batch=16,
        save=True,
        save_json=True,
        project='runs/val',
        name='test_results'
    )
    
    # Print evaluation metrics
    print("\nEvaluation Metrics:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    print(f"Precision: {results.box.precision:.3f}")
    print(f"Recall: {results.box.recall:.3f}")

if __name__ == "__main__":
    model_input = input("Enter model name (e.g., 'yolov8n.pt'): ")
    test_model(model_input)
