"""Run the trained detector on one image and save a copy with the boxes drawn.

    python predict_image.py test_image_1.jpg --weights runs/detect/train/weights/best.pt
"""
import argparse

import config


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('image')
    parser.add_argument('--weights', default=config.WEIGHTS_PATH)
    parser.add_argument('--conf', type=float, default=config.CONFIDENCE)
    parser.add_argument('--output-dir', default='runs/detect')
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    model.predict(source=args.image, save=True, conf=args.conf,
                  project=str(args.output_dir), name='final_predictions', exist_ok=True)
    print(f"Done. The annotated image is in '{args.output_dir}/final_predictions'.")


if __name__ == '__main__':
    main()
