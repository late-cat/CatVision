# 🐱 CatVision

Real-time human motion detection using your webcam.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-red.svg)](https://opencv.org)

---

## ⚡ Features

- Real-time motion detection
- Audio alert on detection
- Works on any webcam
- Simple configuration
- Beginner-friendly code

---

## 📥 Quick Start

```bash
git clone https://github.com/late-cat/CatVision.git
cd CatVision
pip install -r requirements.txt
python app.py
```

Open: http://localhost:5000

---

## 📂 Project Structure

```
CatVision/
├── motion/
│   ├── __init__.py
│   └── detector.py         # Motion detection logic
├── static/
│   ├── css/
│   │   └── style.css       # Styles
│   ├── js/
│   │   └── app.js          # Alert handling
│   └── audio/
│       └── alert.mp3       # Alert sound
├── templates/
│   └── index.html          # Web interface
├── app.py                  # Flask server
├── config.py               # Settings
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
VIDEO_SOURCE = 0            # Camera index (0, 1, 2...)
SERVER_PORT = 5000          # Web server port
MIN_CONTOUR_AREA = 150      # Minimum detection size
MAX_CONTOUR_AREA = 7000     # Maximum detection size
ALERT_DURATION = 2.0        # Alert duration in seconds
```

---

## 🛠️ Tech Stack

| Tech | Purpose |
|------|---------|
| Flask | Web server |
| OpenCV | Video processing |
| NumPy | Array operations |

---

## 📝 How It Works

1. Captures webcam frames
2. Subtracts background to find movement
3. Filters by size and shape
4. Tracks movement speed
5. Triggers alert if human-like

---

## 👨‍💻 Author

**late-cat (BAPI)** — [@late-cat](https://github.com/late-cat)
