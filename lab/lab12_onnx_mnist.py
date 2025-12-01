import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string
import onnxruntime as ort

# -----------------------------
# 1. ONNX Runtime 세션 로드
# -----------------------------
# CPU 전용
session = ort.InferenceSession(
    "tests/mnist.onnx",
    providers=["CPUExecutionProvider"]
)

INPUT_NAME = "image"   # ONNX 입력 이름 (export 시 지정한 이름)
OUTPUT_NAME = "logits" # ONNX 출력 이름

# -----------------------------
# 2. 이미지 전처리 함수
#    - 클라이언트에서 받은 이미지 파일 → [1, 1, 28, 28] float32
# -----------------------------
def preprocess_image(file) -> np.ndarray:
    # 1) 이미지 로드 및 그레이스케일 변환
    image = Image.open(file).convert("L")

    # 2) MNIST와 색상 방향 맞추기 (MNIST → 검정 배경 / 흰 글씨)
    image = ImageOps.invert(image)

    # 3) MNIST 크기(28x28)에 맞게 리사이즈
    image = image.resize((28, 28))

    # 4) numpy 배열 변환 및 0~1 스케일 정규화
    img_array = np.array(image).astype(np.float32) / 255.0

    # 5) 학습 때 사용한 Normalize 값 적용
    mean = 0.1307
    std = 0.3081
    img_array = (img_array - mean) / std

    # 6) [H, W] → [1, 1, H, W] : 배치/채널 차원 추가
    img_array = img_array[np.newaxis, np.newaxis, :, :]

    return img_array



# -----------------------------
#  Flask 앱
# -----------------------------
app = Flask(__name__)


# 테스트용 업로드 폼 
INDEX_HTML = """
<!doctype html>
<title>MNIST ONNX Runtime Demo</title>
<h1>손글씨 숫자 이미지 업로드 (28x28 흑백 권장)</h1>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file>
  <input type=submit value="Predict">
</form>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


# -----------------------------
# 4. 추론 엔드포인트
#    - POST /predict
#    - form-data로 이미지 1장(file) 업로드
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    # 파일 체크
    if "file" not in request.files:
        return jsonify({"error": "file field is required"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        # 1) 전처리: 파일 → [1, 1, 28, 28] float32
        img_array = preprocess_image(file)

        # 2) ONNX Runtime 추론
        ort_inputs = {INPUT_NAME: img_array}
        ort_outputs = session.run([OUTPUT_NAME], ort_inputs)
        logits = ort_outputs[0]          # shape: [1, 10]

        # 3) 예측 결과: argmax
        pred_class = int(np.argmax(logits[0]))

        # (선택) softmax로 확률도 계산
        exp = np.exp(logits[0] - np.max(logits[0]))
        probs = (exp / exp.sum()).tolist()

        return jsonify({
            "pred_class": pred_class,
            "probabilities": probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
