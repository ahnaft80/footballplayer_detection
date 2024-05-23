from ultralytics import YOLO
import os

# Load model
model = YOLO('model/best.pt')

# Define the absolute path for the runs directory
save_dir = '/Users/ahnaftajwarrafid/Documents/projects/yolo/runs/'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Predict and save results in the specified directory
result = model.predict('input/08fd33_0.mp4', save=True, save_dir=save_dir)

print(result[0])
