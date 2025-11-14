🐱 CatVision

A simple & fast real-time human-detection system using your device's webcam.
Powered by Flask + YOLO + OpenCV, runs on any laptop/PC without extra setup.

⚡️ Features
- Runs on any device with a webcam
- Real-time human detection
- Lightweight and beginner-friendly
- No complicated setup (no virtual environment needed)
- Just clone → install → run

🔧 Requirements
- Python 3.9+
- A working webcam
- Git installed on your system

📥 Installation (Beginner Friendly)
Anyone can run your project in 3 steps:

1️⃣ Clone the project

Windows / Mac / Linux — all same command
Open any folder → Right-click → Open Terminal
Then run:

```
git clone https://github.com/late-cat/CatVision.git
```

This creates a folder named `CatVision`.

2️⃣ Install the required packages

Go inside the project folder:

```
cd CatVision
pip install -r requirements.txt
```

This will install:
- Flask
- OpenCV
- NumPy
- Ultralytics (YOLO)
- Gunicorn (not required locally but harmless)

3️⃣ Run the application

Just run:

```
python app.py
```

You will see something like:

```
Running on http://127.0.0.1:5000
```

Open your browser and visit:

- http://localhost:5000
- http://127.0.0.1:5000

Your webcam will turn on and detection begins.

📌 Notes
- Make sure your camera is not used by another app.
- If your device has multiple cameras, you can change the camera index in `app.py`:

```
detector = MotionDetector(video_source=1)
```

- Works on Windows/Mac/Linux.

📂 Project Structure

```
CatVision/
├─ motion/
│  ├─ __init__.py
│  └─ detector.py
├─ static/
│  ├─ style.css
│  └─ app.js
├─ templates/
│  └─ index.html
├─ app.py
├─ requirements.txt
└─ README.txt
```

🙌 Credits
- Created by late-cat(bapi)
- Built with Flask, OpenCV & YOLO
