import os
import json
import random
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from config import logger, MODEL_DIR, DATA_DIR, MODEL_PATH, TFIDF_PATH, SCALER_PATH, METRICS_PATH
from preprocessing import TextPreprocessor
from feature_engineering import FeatureExtractor
from evaluation import train_and_evaluate_models

# Vocabulary lists for synthetic generation
TOPICS = ["Artificial Intelligence", "Healthy Nutrition", "Personal Finance", "Remote Work", "Digital Marketing", "SEO Optimization"]
TRANSITIONS = ["However", "Therefore", "Furthermore", "Consequently", "In addition", "Moreover", "As a result", "On the other hand"]
MISTAKES = ["definately", "seperate", "recieve", "accomodate", "untill", "teh", "goverment", "enviorment", "basicly", "truely"]

def generate_high_quality(topic: str) -> str:
    """Generates a high-quality article template with structured headings, transition words, and links."""
    para1 = f"# Ultimate Guide to {topic}\n\nIn the modern world, {topic} has become increasingly critical for personal and professional growth. {random.choice(TRANSITIONS)}, many people struggle to implement it effectively. Furthermore, understanding the core principles can yield significant advantages. For example, a recent industry survey showed that structured practices lead to a 35% increase in efficiency."
    para2 = f"## Key Strategies for Success\n\nTo achieve the best results in {topic}, we must follow a systematic approach. First, you should define your goals clearly. Second, it is vital to audit your performance regularly. To read more about this, look at our [advanced tutorial](/blog/insights) or visit the [official resource site](https://wikipedia.org/wiki/{topic.replace(' ', '_')}). Integrating these habits will improve your workflow."
    para3 = f"## Conclusion\n\nIn summary, mastering {topic} is a continuous learning journey. Therefore, adopting a disciplined routine is essential. By following these outlined suggestions, you will enhance both your output quality and overall performance."
    return f"{para1}\n\n{para2}\n\n{para3}"

def generate_medium_quality(topic: str) -> str:
    """Generates a medium-quality article template with moderate structure and a few minor spelling errors."""
    para1 = f"Introduction to {topic}\n\n{topic} is very important for everyone nowadays. A lot of people try to do this but they don't know the basic steps. It is basicly simple if you follow standard rules. We will talk about some simple tips. Many businesses are using this enviorment to grow."
    para2 = f"Simple Steps for {topic}\n\nFirst, you need to write down what you want. Second, try to work on it every day. Sometimes it is hard but it is truely rewarding. If you want more details, you should check some online blogs. We recieve feedback that this works for most people."
    para3 = f"Conclusion\n\nSo, {topic} is helpful. If you try, you can get better untill you reach your target. We hope this was a useful overview."
    # Inject a few spelling mistakes
    text = f"{para1}\n\n{para2}\n\n{para3}"
    return text

def generate_low_quality(topic: str) -> str:
    """Generates a low-quality article: spammy, repetitive, poorly structured, with passive voice and high spelling errors."""
    # Repetitive keyword stuffing
    spam_sentences = [
        f"{topic} is best.", f"Buy {topic} services now.", f"We sell cheap {topic}.", 
        f"Get {topic} fast and easy.", f"Click here for {topic} offers.", f"Best {topic} in town."
    ]
    stuffed = " ".join(random.choices(spam_sentences, k=6))
    
    # Rambling passive voice sentences
    passive_sent = "It was decided by teh boss that the work should be done by the workers after it was shown that mistakes were made by the team."
    
    # Incoherent paragraphs with high mistakes
    para1 = f"hello friend read this about {topic}. {stuffed} this is definately teh most import thing. We want to sell you this seperate package."
    para2 = f"It was believed that {topic} was created by engineers. {passive_sent} Many things are recieved untill we accomodate the goverment."
    para3 = f"click now link below. cheap deals. no headings at all."
    
    return f"{para1}\n\n{para2}\n\n{para3}"

def build_dataset(num_samples: int = 1500) -> pd.DataFrame:
    """Generates a balanced dataset of high, medium, and low quality texts."""
    logger.info(f"Generating {num_samples} synthetic articles...")
    data = []
    
    # Balance classes (1/3 each)
    samples_per_class = num_samples // 3
    
    # High Quality (class 2)
    for _ in range(samples_per_class):
        topic = random.choice(TOPICS)
        text = generate_high_quality(topic)
        data.append({"text": text, "label": 2}) # 2 = High
        
    # Medium Quality (class 1)
    for _ in range(samples_per_class):
        topic = random.choice(TOPICS)
        text = generate_medium_quality(topic)
        data.append({"text": text, "label": 1}) # 1 = Medium
        
    # Low Quality (class 0)
    for _ in range(samples_per_class):
        topic = random.choice(TOPICS)
        text = generate_low_quality(topic)
        data.append({"text": text, "label": 0}) # 0 = Low
        
    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save raw dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "synthetic_content.csv", index=False)
    logger.info(f"Dataset saved to {DATA_DIR / 'synthetic_content.csv'}")
    return df

def run_training_pipeline():
    logger.info("Starting training pipeline...")
    
    # 1. Load / Generate Data
    df = build_dataset(num_samples=1500)
    
    # 2. Preprocess & Extract Tabular Features
    preprocessor = TextPreprocessor()
    extractor = FeatureExtractor()
    
    logger.info("Extracting features from articles...")
    tabular_features_list = []
    processed_texts = []
    
    for idx, row in df.iterrows():
        # Preprocess
        proc_data = preprocessor.process_pipeline(row["text"])
        # Extract features
        feats = extractor.extract_features(row["text"], proc_data)
        tabular_features_list.append(feats)
        # Store lemmatized processed text for TF-IDF
        processed_texts.append(proc_data["processed_text"])
        
        if (idx + 1) % 300 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} articles...")
            
    tabular_df = pd.DataFrame(tabular_features_list)
    feature_names = list(tabular_df.columns)
    
    # Convert features dataframe to float matrix
    X_tabular = tabular_df.values
    
    # 3. TF-IDF Feature Extraction
    logger.info("Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(processed_texts).toarray()
    
    tfidf_feature_names = [f"tfidf_{name}" for name in tfidf.get_feature_names_out()]
    all_feature_names = feature_names + tfidf_feature_names
    
    # Combine Tabular + TF-IDF
    X_combined = np.hstack((X_tabular, X_tfidf))
    y = df["label"].values
    
    # 4. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale only the tabular part (first len(feature_names) columns)
    logger.info("Scaling tabular features...")
    scaler = StandardScaler()
    
    X_train_scaled_tabular = scaler.fit_transform(X_train[:, :len(feature_names)])
    X_test_scaled_tabular = scaler.transform(X_test[:, :len(feature_names)])
    
    # Reassemble scaled tabular features + TF-IDF
    X_train_final = np.hstack((X_train_scaled_tabular, X_train[:, len(feature_names):]))
    X_test_final = np.hstack((X_test_scaled_tabular, X_test[:, len(feature_names):]))
    
    # 5. Train & Evaluate Models
    best_name, best_model, comparison_df, metrics_dict = train_and_evaluate_models(
        X_train_final, y_train, X_test_final, y_test, all_feature_names
    )
    
    # 6. Save Artifacts
    logger.info("Saving trained models and scalers...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(tfidf, TFIDF_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save feature names and evaluation metrics
    meta_data = {
        "best_model_name": best_name,
        "feature_names": feature_names,
        "tfidf_feature_names": tfidf_feature_names,
        "all_feature_names": all_feature_names,
        "metrics": metrics_dict,
        "comparison_table": comparison_df.to_dict(orient="records")
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(meta_data, f, indent=4)
        
    logger.info("Training pipeline completed successfully! Saved all artifacts to models/")
    
if __name__ == "__main__":
    run_training_pipeline()
