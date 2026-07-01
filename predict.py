import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, List, Tuple

from config import logger, MODEL_PATH, TFIDF_PATH, SCALER_PATH, METRICS_PATH, QUALITY_LABELS
from preprocessing import TextPreprocessor
from feature_engineering import FeatureExtractor

class ContentPredictor:
    """
    Inference pipeline for ContentIQ: processes input text, performs prediction,
    and calculates explainable SHAP values.
    """
    def __init__(self):
        logger.info("Loading model artifacts...")
        if not (MODEL_PATH.exists() and TFIDF_PATH.exists() and SCALER_PATH.exists()):
            raise FileNotFoundError(
                "Model files are missing. Please run 'python train.py' first to train and save models."
            )
            
        self.model = joblib.load(MODEL_PATH)
        self.tfidf = joblib.load(TFIDF_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        
        # Load feature names metadata
        with open(METRICS_PATH, "r") as f:
            metadata = json.load(f)
            self.feature_names = metadata["feature_names"]
            self.tfidf_feature_names = metadata["tfidf_feature_names"]
            self.all_feature_names = metadata["all_feature_names"]
            
        self.preprocessor = TextPreprocessor()
        self.extractor = FeatureExtractor()
        
        # Initialize SHAP explainer
        try:
            # Tree-based explainer for Random Forest/XGBoost
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Could not initialize TreeExplainer: {e}. Trying generic Explainer.")
            try:
                self.explainer = shap.Explainer(self.model)
            except Exception as ex:
                logger.error(f"Failed to initialize any SHAP explainer: {ex}")
                self.explainer = None

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        Predicts the quality class and confidence score, and extracts local SHAP explanations.
        """
        # 1. Preprocess and Extract Features
        proc_data = self.preprocessor.process_pipeline(raw_text)
        tabular_feats = self.extractor.extract_features(raw_text, proc_data)
        
        # Convert tabular to float array and scale
        X_tab = np.array([list(tabular_feats.values())])
        X_tab_scaled = self.scaler.transform(X_tab)
        
        # TF-IDF vectorization
        X_tf = self.tfidf.transform([proc_data["processed_text"]]).toarray()
        
        # Combine scaled tabular + TF-IDF
        X_combined = np.hstack((X_tab_scaled, X_tf))
        
        # 2. Predict Class and Probability
        probs = self.model.predict_proba(X_combined)[0]
        pred_class_idx = int(np.argmax(probs))
        confidence = float(probs[pred_class_idx])
        pred_label = QUALITY_LABELS[pred_class_idx]
        
        # Compute Overall Quality Score (scaled from probability weights)
        # Class weights: Low Quality (0) -> 20%, Medium Quality (1) -> 60%, High Quality (2) -> 100%
        # Or simpler: Quality Score = (0.1 * prob[Low] + 0.6 * prob[Medium] + 1.0 * prob[High]) * 100
        quality_score = (0.1 * probs[0] + 0.6 * probs[1] + 1.0 * probs[2]) * 100.0
        
        # 3. Explain with SHAP
        shap_explanation = self._get_local_shap_explanation(X_combined, pred_class_idx)
        
        return {
            "prediction": pred_label,
            "class_index": pred_class_idx,
            "confidence": confidence,
            "overall_score": quality_score,
            "features": tabular_feats,
            "shap_explanation": shap_explanation,
            "tokens": proc_data["tokens"],
            "cleaned_text": proc_data["cleaned_text"]
        }

    def _get_local_shap_explanation(self, X_combined: np.ndarray, pred_class_idx: int) -> Dict[str, Any]:
        """Calculates SHAP explanations for the specific prediction class."""
        explanation = {"positive": [], "negative": []}
        
        class_shap = None
        if self.explainer is not None:
            try:
                # Compute shap values for the single sample
                # X_combined shape is (1, n_features)
                shap_values = self.explainer.shap_values(X_combined)
                
                # Extract shap values for the predicted class
                if isinstance(shap_values, list):
                    class_shap = shap_values[pred_class_idx][0]
                elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
                    # Shape (nsamples, nfeatures, nclasses)
                    class_shap = shap_values[0, :, pred_class_idx]
                elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
                    # Binary class or single output shape (nsamples, nfeatures)
                    class_shap = shap_values[0]
                else:
                    class_shap = shap_values[0] if hasattr(shap_values, "__len__") else None
            except Exception as e:
                logger.warning(f"SHAP explanation calculation failed: {e}. Falling back to coefficients/importances.")
                
        # Fallback to model coefficients (for Logistic Regression) or feature importances
        if class_shap is None:
            try:
                if hasattr(self.model, "coef_"):
                    # Linear model coefficient contribution
                    coefs = self.model.coef_[pred_class_idx]
                    class_shap = coefs * X_combined[0]
                elif hasattr(self.model, "feature_importances_"):
                    # Tree model feature importance contribution proxy
                    importances = self.model.feature_importances_
                    # Center importances around median to create positive/negative contributions
                    class_shap = importances * X_combined[0]
                    class_shap = class_shap - np.median(class_shap)
            except Exception as ef:
                logger.error(f"Fallback coefficients contribution extraction failed: {ef}")
                
        try:
            if class_shap is not None:
                # Map SHAP values to feature names
                feature_impacts = []
                for name, val in zip(self.all_feature_names, class_shap):
                    # We format names nicely (e.g. replacing underscores, capital letters)
                    nice_name = name.replace("_", " ").title()
                    feature_impacts.append((nice_name, float(val)))
                    
                # Sort feature impacts
                # Positive impacts push the score TOWARDS the predicted class
                # Negative impacts pull the score AWAY from the predicted class
                feature_impacts.sort(key=lambda x: x[1], reverse=True)
                
                # Filter out TF-IDF sparse features from readable text list for cleaner SaaS explanations,
                # but keep statistical features
                stat_impacts = [x for x in feature_impacts if not x[0].lower().startswith("tfidf")]
                
                positive_factors = [x for x in stat_impacts if x[1] > 0.001][:5]
                negative_factors = [x for x in stat_impacts if x[1] < -0.001]
                negative_factors.sort(key=lambda x: x[1]) # most negative first
                negative_factors = negative_factors[:5]
                
                explanation["positive"] = [{"feature": f[0], "impact": f[1]} for f in positive_factors]
                explanation["negative"] = [{"feature": f[0], "impact": f[1]} for f in negative_factors]
                
                # Include raw impacts for plotly visualization in dashboard
                explanation["all_stat_impacts"] = [{"feature": f[0], "impact": f[1]} for f in stat_impacts]
                
        except Exception as e:
            logger.error(f"Error extracting SHAP explanations: {e}", exc_info=True)
            
        return explanation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run inference on sample text")
    args = parser.parse_args()
    
    if args.sample:
        try:
            predictor = ContentPredictor()
            
            # Simple test cases
            good_text = """
            # Modern Software Architecture Design
            
            Building scalable systems requires a solid understanding of software design patterns. However, it requires a lot of diligence. 
            Furthermore, microservices offer significant benefits. For example, deploying isolated components increases overall reliability.
            
            To understand these principles, refer to our [advanced guides](/design/patterns) or check the [official specs](https://standards.org).
            In summary, modularity leads to cleaner code bases and long-term sustainability.
            """
            
            poor_text = """
            seo seo seo. buy seo. Cheap software architecture fast. we sell teh best services. click now. seperately recieve deals untill tomorrow.
            """
            
            logger.info("\n--- PREDICTING GOOD TEXT ---")
            res_good = predictor.predict(good_text)
            print(f"Prediction: {res_good['prediction']} (Overall Score: {res_good['overall_score']:.1f}%, Confidence: {res_good['confidence']:.2f})")
            print("Positive SHAP factors:")
            for factor in res_good["shap_explanation"]["positive"]:
                print(f"  + {factor['feature']}: {factor['impact']:.4f}")
            print("Negative SHAP factors:")
            for factor in res_good["shap_explanation"]["negative"]:
                print(f"  - {factor['feature']}: {factor['impact']:.4f}")
                
            logger.info("\n--- PREDICTING POOR TEXT ---")
            res_poor = predictor.predict(poor_text)
            print(f"Prediction: {res_poor['prediction']} (Overall Score: {res_poor['overall_score']:.1f}%, Confidence: {res_poor['confidence']:.2f})")
            print("Positive SHAP factors:")
            for factor in res_poor["shap_explanation"]["positive"]:
                print(f"  + {factor['feature']}: {factor['impact']:.4f}")
            print("Negative SHAP factors:")
            for factor in res_poor["shap_explanation"]["negative"]:
                print(f"  - {factor['feature']}: {factor['impact']:.4f}")
                
        except Exception as e:
            print(f"Error testing predictor: {e}")
            print("Please make sure you have run 'python train.py' first.")
