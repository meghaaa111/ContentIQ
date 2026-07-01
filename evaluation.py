import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from config import logger

def calculate_multiclass_roc_auc(model, X_test, y_test, num_classes=3) -> float:
    """Calculates multiclass ROC-AUC score using One-Vs-Rest (ovr) method."""
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            # Handle binary edge cases if generated data only has 2 classes
            if probs.shape[1] < num_classes:
                return 0.5
            return float(roc_auc_score(y_test, probs, multi_class='ovr', average='weighted'))
    except Exception as e:
        logger.warning(f"Failed to calculate ROC-AUC: {e}")
    return 0.0

def train_and_evaluate_models(
    X_train: np.ndarray, y_train: np.ndarray, 
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list
) -> Tuple[str, Any, pd.DataFrame, Dict[str, Any]]:
    """
    Trains Logistic Regression, Random Forest, and XGBoost models.
    Compares their performance on validation/test data and returns 
    the best performing model.
    """
    logger.info("Initializing model training and evaluation pipeline...")
    
    # 1. Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            random_state=42, 
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    }
    
    results = []
    trained_models = {}
    detailed_metrics = {}
    
    # 2. Train and evaluate each model
    for name, clf in models.items():
        logger.info(f"Training {name}...")
        try:
            clf.fit(X_train, y_train)
            trained_models[name] = clf
            
            # Predict
            preds = clf.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='weighted', zero_division=0)
            roc_auc = calculate_multiclass_roc_auc(clf, X_test, y_test)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, preds).tolist()
            
            logger.info(f"{name} Results - Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            })
            
            # Keep detailed stats for UI plotting
            detailed_metrics[name] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "confusion_matrix": cm,
                "predictions": preds.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training {name}: {e}", exc_info=True)
            
    # 3. Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # 4. Determine best model based on F1-Score, prioritizing tree models for Tree SHAP if scores are tied
    preference_order = {"Random Forest": 3, "XGBoost": 2, "Logistic Regression": 1}
    best_model_name = "Random Forest"
    best_f1 = -1.0
    
    for row in results:
        score = row["F1-Score"]
        name = row["Model"]
        if score > best_f1:
            best_f1 = score
            best_model_name = name
        elif score == best_f1:
            # Tie break based on Tree SHAP preference
            if preference_order.get(name, 0) > preference_order.get(best_model_name, 0):
                best_model_name = name
            
    logger.info(f"Best model selected: {best_model_name} with F1-Score: {best_f1:.4f}")
    best_model = trained_models[best_model_name]
    
    return best_model_name, best_model, comparison_df, detailed_metrics

if __name__ == "__main__":
    # Test execution with dummy arrays
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3, n_informative=8, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_name, best_clf, df, details = train_and_evaluate_models(
        X_train, y_train, X_test, y_test, [f"feat_{i}" for i in range(10)]
    )
    print("\nComparison Table:")
    print(df)
