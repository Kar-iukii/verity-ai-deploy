"""Serve the six-score KYC model packaged in this image."""
import hashlib
import json
import logging
import math
from pathlib import Path

from ml.libsvm_runtime import ProbabilityModel
from ml.schema import CLASSES, FEATURES, validate_scores

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parents[1] / 'ml' / 'models'


class ModelNotLoadedError(RuntimeError):
    """The saved classifier or scaler could not be loaded."""


def _load():
    manifest = json.loads((MODEL_DIR / 'artifact_manifest.json').read_text())
    version = (MODEL_DIR / 'model_version.txt').read_text().splitlines()[0]
    if manifest['version'] != version or manifest['features'] != FEATURES or manifest['classes'] != CLASSES:
        raise ValueError('Model version, features or labels differ from the service contract')
    for name in ('face_review.model', 'scaler.json'):
        if hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest() != manifest['artifacts'][name]:
            raise ValueError(f'Artifact checksum mismatch for {name}')
    scaler = json.loads((MODEL_DIR / 'scaler.json').read_text())
    if scaler['model_version'] != version or scaler['features'] != FEATURES or scaler['classes'] != CLASSES:
        raise ValueError('Scaler does not belong to this model')
    minimum, maximum = scaler['min'], scaler['max']
    if len(minimum) != len(FEATURES) or len(maximum) != len(FEATURES):
        raise ValueError('Invalid scaler dimensions')
    if any(not (math.isfinite(lo) and math.isfinite(hi) and lo < hi) for lo, hi in zip(minimum, maximum)):
        raise ValueError('Invalid scaler ranges')
    return version, minimum, maximum, ProbabilityModel(MODEL_DIR / 'face_review.model', CLASSES)


try:
    version, minimum, maximum, model = _load()
    model_loaded = True
except Exception:
    logger.exception('Could not load KYC model')
    version, model_loaded = 'unknown', False


def get_model_version():
    return version


def get_model_status():
    return {'model_loaded': model_loaded, 'model_version': version, 'trained_at': None}


def predict_kyc_decision(scores):
    if not model_loaded:
        raise ModelNotLoadedError('SVM model is not loaded')
    scores, errors = validate_scores(scores)
    if errors:
        raise ValueError('; '.join(errors))
    scaled = [min(1., max(0., (scores[name] - lo) / (hi - lo)))
              for name, lo, hi in zip(FEATURES, minimum, maximum)]
    decision, probabilities = model.predict(scaled)
    return {'finalDecision': decision, 'probabilities': probabilities,
            'verifiedProbability': probabilities['Verified'],
            'reviewProbability': probabilities['Manual Review'],
            'rejectedProbability': probabilities['Rejected'],
            'decisionBasis': 'svm_model', 'modelVersion': version,
            'decisionPolicy': {'name': 'probability_argmax'}}
