import numpy as np
from app import app
from services import faceMatchService as face, livenessService as live, svmService as svm
from services.ocrService import get_reader

def validate():
    face.ensure_model_ready()
    live.ensure_model_ready()
    if not svm.model_loaded:
        raise RuntimeError("SVM model failed to load")
    get_reader()
    # Validate both anti-spoof networks without using a real person's image.
    from pathlib import Path
    for path in Path(live.MODEL_DIR).glob("*.pth"):
        live._model.predict(np.zeros((80, 80, 3), dtype=np.uint8), str(path))
    print("All four model components loaded successfully", flush=True)

def validate_decision_routes():
    # Exercise the same request/response contract used by the Node backend.
    from ml.schema import LEGACY_FEATURES
    client = app.test_client()
    status = client.get('/api/svm/status')
    if status.status_code != 200 or status.get_json()['model_version'] != '3.1.1':
        raise RuntimeError("Expected the v3.1.1 model in this image")
    for face_score, expected in [(5, 'rejected'), (37, 'review'), (80, 'verified')]:
        response = client.post('/api/svm/predict', json=dict(zip(
            LEGACY_FEATURES, [face_score, 1, 1, 80, 150, 45])))
        if response.status_code != 200 or response.get_json()['finalDecision'] != expected:
            raise RuntimeError("KYC model prediction smoke test failed")


if __name__ == "__main__":
    validate()
    validate_decision_routes()
