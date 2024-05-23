# Soccer Player Detection

This project focuses on detecting soccer players, goalkeepers, referees, and soccer balls in images and videos using the YOLOv5 object detection model. The dataset used for training includes images of soccer players and soccer balls, sourced from Roboflow.

## Project Overview

The project involves the following key steps:
1. **Dataset Preparation**: Downloading the dataset from Roboflow.
2. **Model Training**: Training the YOLOv5 model on the dataset using Google Colab with GPU support.
3. **Evaluation**: Evaluating the model's performance and iteratively improving the detection accuracy.
4. **Custom Tracker Implementation**: Implementing a custom tracker for enhanced object tracking.

### Final Output
To see the final output, check the `output` folder or the `runs` directory.

## Dataset

The project utilizes the following dataset:
1. **Football Players Detection**: Includes images of soccer players, goalkeepers, and referees.

The dataset was sourced from Roboflow and structured for training.

## Installation

### Prerequisites

- Python 3.x
- Git
- Google Colab (optional, for training with GPU)

### Dependencies

Install the required dependencies:

```sh
pip install torch torchvision torchaudio
pip install ultralytics
pip install roboflow
```

### Cloning the Repository

Clone the repository to your local machine:

```sh
git clone https://github.com/your-username/new-repo-name.git
cd new-repo-name
```

## Dataset Preparation

1. **Download Dataset from Roboflow**:

```python
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="6Qa5tKMsuA7U46cDC2zg")

# Football Players Detection Dataset
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov5")
```

## Training the Model

The notebook located at `training/player_training.ipynb` contains the code to generate the model. To train the YOLOv5 model, use the following command in your Google Colab or local environment with GPU support:

```python
!yolo task=detect mode=train model=yolov5x.pt data=/content/football-players-detection-1/data.yaml epochs=100 imgsz=640 batch=16 cache=True
```

Note: The model is not included in the repository. You have to generate the model by running the training notebook and saving the best model to `model/best.pt`.

## Running Inference

### YOLO Inference

To run inference using the trained YOLO model, use the following code:

```python
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
```

Ensure you create a folder named `model` and place the trained model (`best.pt`) inside it. The output will be saved in the `runs/detect` directory.

### Custom Tracker Implementation

To run the custom tracker and see the results, run the `main.py` script and check the output in the `output` folder:

```python
from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np

def main():
    video_frames = read_video('input/08fd33_0.mp4')
    tracker = Tracker('model/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output/output_video.avi')

if __name__ == '__main__':
    main()
```

Run `main.py` to see the custom tracker in action. Ensure the model is placed in the `model/best.pt` directory.

## Pretrained Models and Outputs

- **Predict Folder**: Contains the output of the general pretrained YOLOv8 model.
- **Predict2 Folder**: Contains the output of the best model trained from the Roboflow dataset.

Check these folders for sample outputs.

## Contributing

If you want to contribute to this project, feel free to fork the repository and submit pull requests. Contributions are welcome!

## License

This project is licensed under the [MIT License](LICENSE).

