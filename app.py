from flask import Flask, render_template, Response, jsonify, send_file
from motion.detector import MotionDetector
import atexit


app = Flask(__name__)
detector = MotionDetector(video_source=0)
atexit.register(detector.release)


def generate():
    while True:
        frame_bytes, _, _ = detector.next_frame()
        if frame_bytes is None:
            break
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/alert_status")
def alert_status():
    return jsonify({"alert": detector.is_alert_active()})

@app.route("/alert.mp3")
def alert_mp3():
    return send_file("alert.mp3", mimetype="audio/mpeg")

@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)