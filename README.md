# Plant Disease Detection System

A comprehensive machine learning system for detecting and classifying plant diseases using computer vision and deep learning techniques. This project implements multiple approaches including CNN-based classification and YOLO-style object detection for real-time plant disease identification.

## 🌱 Project Overview

This project provides a complete solution for plant disease detection with the following capabilities:

- **Multi-class Disease Classification**: Detects 38 different plant diseases across various crops
- **Real-time Detection**: Live camera feed analysis with bounding box detection
- **Mobile-Optimized**: TensorFlow Lite models for deployment on edge devices
- **Comprehensive Dataset**: Uses the New Plant Diseases Dataset (Augmented) with 54,305+ images
- **Multiple Implementation Approaches**: CNN classification, YOLO detection, and bounding box analysis

## 📁 Project Structure

```
plant/
├── m.py                          # Main training script (MobileNetV2)
├── plant_test.py                 # Model testing and evaluation
├── plant_yolo.py                 # Real-time YOLO-style detection
├── plant_2.py                    # Alternative model implementation
├── plant_5_bounding.py           # Bounding box detection implementation
├── check.py                      # GPU availability checker
├── plant_disease_model.tflite    # Optimized TensorFlow Lite model
├── download.jpg                  # Sample image
├── New Plant Diseases Dataset(Augmented)/  # Training dataset
│   └── New Plant Diseases Dataset(Augmented)/
│       ├── train/                # Training images (38 classes)
│       └── valid/                # Validation images (38 classes)
├── test/                         # Test images
└── plant affect detection/       # Additional dataset organization
```

## 🎯 Supported Plant Diseases

The system can detect diseases across multiple plant categories:

### Fruits
- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy
- **Blueberry**: Healthy
- **Cherry**: Healthy, Powdery Mildew
- **Grape**: Black Rot, Esca (Black Measles), Healthy, Leaf Blight
- **Peach**: Bacterial Spot, Healthy
- **Strawberry**: Healthy, Leaf Scorch

### Vegetables
- **Corn (Maize)**: Cercospora Leaf Spot, Common Rust, Healthy, Northern Leaf Blight
- **Pepper**: Bacterial Spot, Healthy
- **Potato**: Early Blight, Healthy, Late Blight
- **Tomato**: Bacterial Spot, Early Blight, Healthy, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Tomato Mosaic Virus, Tomato Yellow Leaf Curl Virus

### Other Crops
- **Orange**: Huanglongbing (Citrus Greening)
- **Raspberry**: Healthy
- **Soybean**: Healthy
- **Squash**: Powdery Mildew

## 🚀 Features

### 1. Deep Learning Model
- **Architecture**: MobileNetV2 with transfer learning
- **Input Size**: 128x128 pixels
- **Classes**: 38 different plant disease categories
- **Optimization**: TensorFlow Lite quantization for mobile deployment

### 2. Real-time Detection
- Live camera feed processing
- Bounding box detection for affected areas
- Real-time disease classification
- Confidence score display

### 3. Data Augmentation
- Rotation, scaling, and flipping
- Color and brightness adjustments
- Shear and zoom transformations
- Horizontal flipping

### 4. Model Optimization
- TensorFlow Lite conversion
- Quantization for reduced model size
- Mobile-friendly architecture
- Efficient inference on edge devices

## 📋 Requirements

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, for faster training)
- Webcam (for real-time detection)

### Python Dependencies
```bash
tensorflow>=2.8.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.0.0
tkinter (usually included with Python)
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd plant
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow opencv-python numpy Pillow
   ```

3. **Download the dataset**
   - The project uses the "New Plant Diseases Dataset (Augmented)"
   - Ensure the dataset is properly organized in the specified directory structure

4. **Check GPU availability** (optional)
   ```bash
   python check.py
   ```

## 🎮 Usage

### 1. Training the Model
```bash
python m.py
```
This will:
- Load and preprocess the dataset
- Train a MobileNetV2 model
- Save the trained model
- Convert to TensorFlow Lite format

### 2. Real-time Detection
```bash
python plant_yolo.py
```
This will:
- Open your webcam
- Display real-time disease detection
- Show bounding boxes around affected areas
- Display predicted disease class

### 3. Model Testing
```bash
python plant_test.py
```
This will:
- Load the trained model
- Evaluate on validation data
- Display accuracy metrics

### 4. Bounding Box Detection
```bash
python plant_5_bounding.py
```
This will:
- Perform object detection on images
- Draw bounding boxes around diseased areas
- Save annotated images

## 📊 Model Performance

- **Architecture**: MobileNetV2 with transfer learning
- **Input Resolution**: 128x128 pixels
- **Number of Classes**: 38
- **Training Data**: ~54,305 images
- **Validation Data**: ~13,000 images
- **Model Size**: ~13MB (optimized TFLite)

## 🔧 Configuration

### Model Parameters
- `img_height = 128`
- `img_width = 128`
- `batch_size = 32`
- `num_classes = 38`
- `epochs = 10`

### Data Augmentation Settings
```python
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

## 📈 Training Process

1. **Data Loading**: Images are loaded from the organized dataset structure
2. **Preprocessing**: Images are resized to 128x128 and normalized
3. **Augmentation**: Training data is augmented to improve generalization
4. **Model Training**: MobileNetV2 is fine-tuned on the plant disease dataset
5. **Evaluation**: Model performance is evaluated on validation data
6. **Export**: Model is converted to TensorFlow Lite for deployment

## 🎯 Applications

- **Agricultural Monitoring**: Automated disease detection in farms
- **Plant Health Assessment**: Quick health checks for gardeners
- **Research**: Plant pathology research and data collection
- **Mobile Apps**: Integration into smartphone applications
- **IoT Devices**: Deployment on edge devices for continuous monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: New Plant Diseases Dataset (Augmented) from Kaggle
- **Model Architecture**: MobileNetV2 by Google Research
- **Framework**: TensorFlow and TensorFlow Lite
- **Computer Vision**: OpenCV

## 📞 Support

For questions, issues, or contributions, please:
1. Check the existing issues
2. Create a new issue with detailed description
3. Contact the maintainers

---

**Note**: This project is designed for educational and research purposes. For commercial applications, please ensure compliance with relevant regulations and obtain necessary permissions.
