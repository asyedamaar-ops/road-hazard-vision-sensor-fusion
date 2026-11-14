"""Step 3: fine-tune YOLOv8n on the road-hazard dataset.

    python train_model.py --data path/to/data.yaml

The dataset is the Roboflow export "Sinkhole_Vision_Fusion" v1 (187 images,
classes Concentric_Crack and Pothole). train_and_evaluate.ipynb does the same
run on Colab and also evaluates on the test split.
"""
import argparse

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', default=config.DATASET_YAML, help='path to the dataset data.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None, help='for example 0 for the first GPU, or cpu')
    parser.add_argument('--workers', type=int, default=2, help='2 keeps memory stable on Windows')
    args = parser.parse_args()

    from ultralytics import YOLO      # imported here so --help works without the package

    model = YOLO('yolov8n.pt')
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz,
                device=args.device, workers=args.workers)
    print("Training complete. Weights are in 'runs/detect/train/weights'.")


# the __main__ guard is required on Windows, where data loading starts new processes
if __name__ == '__main__':
    main()
