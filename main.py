"""Flask gateway that exposes YOLO-World detections over HTTP."""

from __future__ import annotations

import base64
from typing import List

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request, render_template_string
from flasgger import Swagger, swag_from
from ultralytics import YOLOWorld


def _load_image_from_url(url: str) -> np.ndarray:
    """Download an image from a URL and decode it into an OpenCV BGR array."""

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise ValueError(f"Không thể tải ảnh từ URL: {exc}") from exc

    image_array = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Không thể giải mã ảnh từ dữ liệu tải về.")

    return image


app = Flask(__name__)

# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
}

swagger = Swagger(app, config=swagger_config)

# Load the YOLO-World model once at startup to avoid repeated warm-up cost.
model = YOLOWorld("yolov8s-world.pt")


@app.route("/")
def home():
    """Home page showing project status."""
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>VisionMark-Img Object Detection API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .status {
                background: #27ae60;
                color: white;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                margin: 20px 0;
                font-weight: bold;
            }
            .links {
                margin: 30px 0;
                text-align: center;
            }
            .link {
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 12px 24px;
                text-decoration: none;
                border-radius: 5px;
                margin: 0 10px;
                transition: background 0.3s;
            }
            .link:hover {
                background: #2980b9;
            }
            .api-info {
                background: #ecf0f1;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            }
            .endpoint {
                background: #f8f9fa;
                padding: 10px;
                margin: 10px 0;
                border-left: 4px solid #3498db;
                font-family: monospace;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 VisionMark-Img Object Detection API</h1>
            <div class="status">✅ Server is running successfully!</div>

            <div class="api-info">
                <h2>📋 API Information</h2>
                <p><strong>Framework:</strong> Flask with YOLO-World (Ultralytics)</p>
                <p><strong>Model:</strong> yolov8s-world.pt</p>
                <p><strong>Status:</strong> Active and ready for requests</p>
            </div>

            <div class="links">
                <a href="/docs/" class="link">📖 Swagger UI Documentation</a>
                <a href="/test" class="link">🧪 Test Interface</a>
            </div>

            <div class="api-info">
                <h2>🔗 Available Endpoints</h2>
                <div class="endpoint">
                    <strong>POST /detect</strong><br>
                    Detect objects in images via URL<br>
                    <em>Parameters:</em> class_name (string/array), image_url (string)<br>
                    <em>Response:</em> count, classes, annotated image (base64)
                </div>
                <div class="endpoint">
                    <strong>GET /test</strong><br>
                    Interactive test interface for the detection API
                </div>
                <div class="endpoint">
                    <strong>GET /docs/</strong><br>
                    Swagger UI documentation for all endpoints
                </div>
            </div>

            <div class="api-info">
                <h2>💡 Usage Example</h2>
                <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST http://localhost:5001/detect \
  -H "Content-Type: application/json" \
  -d '{"class_name": "person", "image_url": "https://example.com/image.jpg"}'</pre>
            </div>
        </div>
    </body>
    </html>
    """)


@app.post("/detect")
@swag_from({
    "tags": ["Object Detection"],
    "summary": "Detect objects in image",
    "description": "Detect specified objects in an image from URL and return count with annotated image",
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": ["string", "array"],
                        "items": {"type": "string"},
                        "example": "person",
                        "description": "Object class(es) to detect (string or array of strings)"
                    },
                    "image_url": {
                        "type": "string",
                        "example": "https://picsum.photos/640/480",
                        "description": "URL of the image to analyze"
                    }
                },
                "required": ["class_name", "image_url"]
            }
        }
    ],
    "responses": {
        "200": {
            "description": "Successful detection",
            "schema": {
                "type": "object",
                "properties": {
                    "classes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of classes that were searched for"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of detected objects"
                    },
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded PNG image with bounding boxes"
                    }
                }
            }
        },
        "400": {
            "description": "Bad request - missing or invalid parameters",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"}
                }
            }
        },
        "500": {
            "description": "Internal server error",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string"}
                }
            }
        }
    }
})
def detect() -> tuple[dict, int]:
    """Receive an image URL and class name(s), return detections and annotated image."""

    payload = request.get_json(silent=True) or {}

    class_field = payload.get("class_name")
    image_url = payload.get("image_url")

    if not image_url:
        return jsonify({"error": "Thiếu 'image_url' trong request body."}), 400

    if not class_field:
        return jsonify({"error": "Thiếu 'class_name' trong request body."}), 400

    if isinstance(class_field, str):
        target_classes: List[str] = [class_field]
    elif isinstance(class_field, list) and all(isinstance(item, str) for item in class_field):
        target_classes = class_field
    else:
        return jsonify({"error": "'class_name' phải là chuỗi hoặc danh sách chuỗi."}), 400

    try:
        image = _load_image_from_url(image_url)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Configure the model to only look for the requested classes.
    model.set_classes(target_classes)

    # Run prediction on the in-memory image.
    results = model.predict(image, verbose=False)

    total_detections = sum(len(result.boxes) for result in results)

    if results:
        annotated_image = results[0].plot()  # YOLO returns one result per input image
    else:
        annotated_image = image

    success, buffer = cv2.imencode(".png", annotated_image)
    if not success:
        return jsonify({"error": "Không thể mã hóa ảnh kết quả."}), 500

    image_base64 = base64.b64encode(buffer).decode("utf-8")

    response_body = {
        "classes": target_classes,
        "count": total_detections,
        "image_base64": image_base64,
    }

    return jsonify(response_body), 200


if __name__ == "__main__":
    # Expose the Flask app for local testing; in production use a proper WSGI server
    app.run(host="0.0.0.0", port=5001, debug=False)

