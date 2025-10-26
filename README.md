# FscanX: YOLO and NIR Scanner Fusion for Fruit Freshness Detection

FscanX is an intelligent fruit freshness detection system that combines YOLO (You Only Look Once) object detection with computer vision techniques to automatically identify and classify the freshness state of fruits. The system can detect multiple fruit types (apples, bananas) and determine whether they are fresh or rotten.

<p align="center">
<img src="train/results.png" width="600" alt="Training Results"><br>
<i>Training performance metrics</i>
</p>

## 🎯 Features

- **Multi-Class Freshness Detection**: Identify fresh and rotten states for apples and bananas
- **YOLO11-Based Model**: Utilizes the latest YOLO11 architecture for fast and accurate detection
- **Real-Time Inference**: Run detection on images, videos, or live camera feeds
- **Flexible Input Sources**: Support for images, image folders, video files, USB cameras, and Raspberry Pi cameras
- **High Accuracy**: Trained on diverse dataset with robust preprocessing and data augmentation
- **Production Ready**: Includes deployment scripts and example implementation

## 📊 Model Performance

The model was trained using **YOLO11s** architecture with the following specifications:
- **Epochs**: 60
- **Resolution**: 640x640
- **Batch Size**: 16
- **Model Size**: YOLO11 Small (YOLO11s.pt)

### Detection Classes
- Fresh Apple
- Fresh Banana  
- Rotten Apple
- Rotten Banana

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- CUDA-capable GPU (recommended for faster inference)
- Windows, Linux, or macOS

### Setup Instructions

1. **Clone the repository and navigate to the project directory:**
```bash
cd FscanX
```

2. **Create and activate a virtual environment:**
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install opencv-python>=4.5.0 numpy>=1.20.0 ultralytics>=8.0.0
```

## 💻 Usage

### Quick Start

Run detection on an image:
```bash
python yolo_detect.py --model train/weights/best.pt --source image.jpg --thresh 0.5
```

Run detection on a video:
```bash
python yolo_detect.py --model train/weights/best.pt --source video.mp4 --thresh 0.5
```

Run detection with USB camera (live feed):
```bash
python yolo_detect.py --model train/weights/best.pt --source usb0 --resolution 1280x720 --thresh 0.5
```

### Command Line Arguments

- `--model`: Path to YOLO model file (required)
  - Example: `"train/weights/best.pt"`
  
- `--source`: Input source (required)
  - Image file: `"image.jpg"`
  - Image folder: `"test_dir"`
  - Video file: `"video.mp4"`
  - USB camera: `"usb0"` (for camera index 0)
  - Raspberry Pi camera: `"picamera0"`

- `--thresh`: Minimum confidence threshold (default: 0.5)
  - Example: `0.4`

- `--resolution`: Display resolution in WxH (optional)
  - Example: `"640x480"` or `"1280x720"`

- `--record`: Record results from video/camera to file
  - Requires `--resolution` to be specified

### Examples

**Detect fruits in a folder of images:**
```bash
python yolo_detect.py --model train/weights/best.pt --source ./test_images --thresh 0.4
```

**Run live detection from webcam with recording:**
```bash
python yolo_detect.py --model train/weights/best.pt --source usb0 --resolution 1280x720 --thresh 0.5 --record
```

**Process a video with custom resolution:**
```bash
python yolo_detect.py --model train/weights/best.pt --source video.mp4 --resolution 1920x1080 --thresh 0.6
```

### Controls During Inference
- **Press 'q'**: Quit the program
- **Press 's'**: Pause/unpause inference  
- **Press 'p'**: Save current frame as `capture.png`

## 🧪 Model Training

The model was trained on a custom dataset using YOLO11. To train your own model or retrain with additional data:

### Using Google Colab

A complete training notebook (`FscanX.ipynb`) is provided for training in Google Colab with GPU support.

1. Upload your dataset in the required YOLO format
2. Upload `FscanX.ipynb` to Google Colab
3. Follow the notebook instructions to:
   - Install Ultralytics
   - Configure training parameters
   - Train the model
   - Export the trained weights

### Training Configuration

The training was conducted with the following parameters (stored in `train/args.yaml`):
```yaml
epochs: 60
batch: 16
imgsz: 640
model: yolo11s.pt
lr0: 0.01
box: 7.5
cls: 0.5
```

### Training Results

Results and metrics are available in the `train/` directory:
- `results.png`: Training curves (precision, recall, mAP)
- `confusion_matrix.png`: Confusion matrix
- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last epoch weights

## 📁 Project Structure

```
FscanX/
├── venv/                    # Virtual environment (gitignored)
├── train/                   # Training outputs and results
│   ├── weights/
│   │   ├── best.pt         # Best trained model
│   │   └── last.pt         # Last checkpoint
│   ├── results.png         # Training metrics visualization
│   ├── confusion_matrix.png
│   └── args.yaml           # Training configuration
├── yolo_detect.py          # Main detection script
├── FscanX.ipynb            # Colab training notebook
├── requirements.txt        # Python dependencies
├── my_model.pt            # Exported model (optional)
├── get-pip.py             # Pip installer script
└── README.md              # This file
```

## 🔧 Dependencies

Core dependencies:
- **OpenCV** (cv2): Image processing and display
- **NumPy**: Array operations
- **Ultralytics**: YOLO model framework
- **PyTorch**: Deep learning backend (installed automatically with Ultralytics)

## 🌐 Deployment

### On PC (Windows/Linux/macOS)

The model can run on any Python environment with the required dependencies.

### On Raspberry Pi

Convert the model to optimized formats like TFLite or ONNX for edge deployment:

```python
from ultralytics import YOLO

model = YOLO('train/weights/best.pt')
model.export(format='onnx')  # Export to ONNX format
```

Then use ONNX Runtime or TensorFlow Lite for inference on Raspberry Pi.

## 📈 Performance Metrics

Training metrics for the current model:
- **Precision (B)**: Box detection precision
- **Recall (B)**: Box detection recall  
- **mAP50 (B)**: Mean Average Precision at IoU=0.50
- **mAP50-95 (B)**: mAP averaged over IoU thresholds 0.50-0.95

View detailed results in `train/results.png` and `train/results.csv`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLO framework
- **Training Notebook**: Based on Evan Juras's YOLO training tutorial
- **Dataset**: Custom fruit freshness detection dataset

## 📧 Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: The virtual environment (`venv/`) is excluded from version control. Make sure to create your own virtual environment before running the scripts.