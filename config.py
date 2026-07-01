import os
import logging
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
PAGES_DIR = BASE_DIR / "pages"

# Create directories if they do not exist
for directory in [MODEL_DIR, DATA_DIR, ASSETS_DIR, PAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Saved Model Paths
MODEL_PATH = MODEL_DIR / "best_model.joblib"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = MODEL_DIR / "feature_scaler.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# Quality Label Mapping
QUALITY_LABELS = {
    0: "Low Quality",
    1: "Medium Quality",
    2: "High Quality"
}

# Feature Extraction Config
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "consequently", "moreover", 
    "additionally", "meanwhile", "nevertheless", "nonetheless", "similarly",
    "subsequently", "likewise", "specifically", "accordingly", "resultantly",
    "hence", "thus", "conversely", "instead", "otherwise", "besides", 
    "meanwhile", "comparatively", "specifically", "ultimately", "essentially"
}

# SEO Thresholds
MIN_RECOMMENDED_WORDS = 300
MAX_RECOMMENDED_WORDS = 2500
IDEAL_KEYWORD_DENSITY_MIN = 0.5   # in percent
IDEAL_KEYWORD_DENSITY_MAX = 3.0   # in percent
IDEAL_PASSIVE_VOICE_MAX = 15.0    # in percent
IDEAL_READABILITY_FLESCH_MIN = 60.0 # Standard Flesch Reading Ease
IDEAL_LEXICAL_DIVERSITY_MIN = 0.40 # Type-Token Ratio
MIN_HEADING_DENSITY = 250         # 1 heading per 250 words is standard

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "contentiq.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("ContentIQ")
