from ultralytics import YOLO
import torch

#  THE WINDOWS FIX: This prevents the multiprocessing crash 
if __name__ == '__main__':

    print(" Starting Day 5: YOLOv8 GPU Training Pipeline...")
    print(f" NVIDIA GPU Detected: {torch.cuda.is_available()}")

    # We already downloaded the data, so we just point YOLO directly to your folder!
    dataset_path = r"D:\Sinkhole_Detection_Project\Sinkhole_Vision_Fusion-1\data.yaml"  # Update this path to your dataset YAML file

    # INITIALIZE THE YOLO BRAIN
    model = YOLO('yolov8n.pt') 

    # START TRAINING
    print("\n Beginning Model Training on the GPU...")
    results = model.train(
        data=dataset_path,
        epochs=50,
        imgsz=640,
        device=0,
        workers=2  # Added this to keep Windows memory stable!
    )

    print("\n TRAINING COMPLETE! The weights are saved in the 'runs/detect/train/weights' folder!")