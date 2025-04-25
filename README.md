# Traffic Sign Detection Using YOLOv8

## Overview
The **Traffic Sign Detection System** is a **Streamlit-based web application** designed for detecting and analyzing traffic signs in images. It utilizes **YOLOv8** for object detection and provides a user-friendly interface with real-time results and advanced analytics.

## Features
- **Image Upload and Analysis**: Users can upload traffic sign images (JPG, JPEG, PNG) and adjust the confidence threshold.
- **Detection and Visualization**: Displays bounding boxes, confidence scores, and detection locations.
- **Advanced Analytics**: Provides detection statistics and visualizations.
- **Custom UI Design**: Intuitive interface with interactive elements.
- **Session Tracking**: Logs analysis history and detection counts.
- **Help & Instructions**: Built-in guide for users.

## Technology Stack
- **YOLOv8** for object detection
- **Streamlit** for web interface
- **OpenCV** for image processing
- **Python** for backend logic

## Installation
### Prerequisites
Ensure you have Python installed. Then install the required dependencies:
```bash
pip install ultralytics streamlit opencv-python-headless pandas numpy matplotlib seaborn
```

### Clone Repository
```bash
git clone https://github.com/muhammadrizwan11/Traffic-Sign-Detection-Using-Yolov8.git
cd Traffic-Sign-Detection-Using-Yolov8
```

### Run the Application
```bash
streamlit run app.py
```

## Dataset
The dataset used for training is sourced from [Kaggle](https://www.kaggle.com/). The model was trained on **Kaggle Notebook** and then deployed in the application.

## Usage
1. **Upload an image** containing traffic signs.
2. **Adjust confidence threshold** if needed.
3. **View detection results** with bounding boxes and confidence scores.
4. **Analyze visualizations** (bar graphs, pie charts, and statistics).

## Results
The trained YOLOv8 model provides **high accuracy and efficient detection** of multiple traffic signs in real-world images.

## Deployment
The application is deployed on **Hugging Face Spaces** for easy accessibility.

## Repository Link
[Traffic Sign Detection Using YOLOv8](https://github.com/muhammadrizwan11/Traffic-Sign-Detection-Using-Yolov8)

## Licens
This project is licensed under the MIT License.

## Contact
For any inquiries, feel free to reach out:
- **GitHub**: [muhammadrizwan11](https://github.com/muhammadrizwan11)
- **Email**: *rizwan.ai.engineer@gmail.com*

---
This README provides all essential details for your project. Let me know if you want any modifications!

