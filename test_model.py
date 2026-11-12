from ultralytics import YOLO

print(" Loading the trained AI brain...")

#  Double-check this path! I copied it straight from your terminal logs 
model_path = r"E:\Vehicle-Damage-Detection\runs\detect\train4\weights\best.pt"

# Load your custom trained model
model = YOLO(model_path)

print(" Feeding it the test image...")

# Run the AI on your new image!
# save=True tells YOLO to save a copy of the image with the boxes drawn on it
results = model.predict(
    source="test_image_1.jpg", 
    save=True, 
    conf=0.5,
    project=r"E:\Vehicle-Damage-Detection\runs\detect",  # <-- THIS IS THE MAGIC LINE!
    name="final_predictions",                            # The folder it will create
    exist_ok=True                                        # Stops it from making predict5, predict6...
)

print("\n DONE! Go check the 'runs/detect/predict' folder to see the AI's drawing!")