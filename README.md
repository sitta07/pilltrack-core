# PillTrack Core (Inspection System)
Verify authenticity of tablets and packaging using AI (Computer Vision) developed with Python, PyQt6 and Deep Learning (YOLOv8 + custom classifier) ​​models.
![Status](https://img.shields.io/badge/Status-Development-orange) 
![Python](https://img.shields.io/badge/Python-3.9%2B-blue) 
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)

## Project Structure
pilltrack-core/
├── config/             # Configuration files (settings.py)
├── core/               # Main Logic
│   ├── camera.py       # Camera Handler (Zoom, Focus)
│   ├── classifier.py   # SmartClassifier (Dual Model Router)
│   ├── detector.py     # YOLO Object Detection
│   ├── processor.py    # Image Processing (Cutout, Filters)
│   └── architecture.py # Model Architecture (ArcFace/Backbone)
├── models/             # AI Models (.pt, .pth, .json)
├── ui/                 # User Interface (PyQt6)
│   └── station_window.py
├── utils/              # Helper functions
├── main.py             # Entry Point
└── requirements.txt    # Dependencies