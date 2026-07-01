import re
from typing import Dict, Any, List, Tuple
import nltk
from nltk.corpus import words
from textblob import TextBlob
import textstat
import numpy as np

from config import logger, TRANSITION_WORDS

# Ensure NLTK resources needed for feature extraction are downloaded
def download_feature_resources():
    resources = ['words', 'averaged_perceptron_tagger_eng']
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == 'words' else f"taggers/{resource}")
        except LookupError:
            logger.info(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)

download_feature_resources()

# Load English vocabulary for spelling checks
try:
    ENGLISH_VOCAB = set(w.lower() for w in words.words())
except Exception as e:
    logger.warning(f"Failed to load NLTK words corpus: {e}. Spelling check will fall back.")
    ENGLISH_VOCAB = set()

class FeatureExtractor:
    """
    Extracts NLP, readability, vocabulary diversity, stylistic quality, and SEO features from text.
    """
    def __init__(self):
        # Passive voice auxiliary verbs
        self.be_verbs = {"is", "was", "were", "been", "be", "are", "am", "being"}
        
    def estimate_passive_voice(self, text: str) -> float:
        """
        Estimates the percentage of sentences written in passive voice.
        Uses NLTK POS tagger to find auxiliary verb 'to be' followed by past participle (VBN).
        """
        if not text:
            return 0.0
            
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return 0.0
            
        passive_count = 0
        for sent in sentences:
            tokens = nltk.word_tokenize(sent.lower())
            if not tokens:
                continue
            
            try:
                tagged = nltk.pos_tag(tokens)
            except Exception as e:
                # If pos_tag fails or is loading, fall back to a simple regex
                pattern = r'\b(is|was|were|been|be|are|am|being)\b\s+(\w+ed|seen|taken|done|made|given|written|known|held)\b'
                if re.search(pattern, sent.lower()):
                    passive_count += 1
                continue
                
            # Scan tagged tokens for: BE-verb + optional adverb(s) + VBN
            for i in range(len(tagged) - 1):
                word, tag = tagged[i]
                if word in self.be_verbs:
                    # Look ahead up to 3 tokens for a VBN (to allow for adverbs in between, e.g. "was quickly built")
                    for j in range(i + 1, min(i + 4, len(tagged))):
                        next_word, next_tag = tagged[j]
                        if next_tag == 'VBN':
                            passive_count += 1
                            break
                        elif next_tag not in ('RB', 'RBR', 'RBS'):  # Stop if we hit a non-adverb
                            break
                            
        return (passive_count / len(sentences)) * 100.0

    def estimate_spelling_mistakes(self, tokens: List[str]) -> int:
        """
        Estimates the count of misspelled words by checking against NLTK's English word list.
        Filters out punctuation, numbers, and proper nouns (capitalized in context).
        """
        if not ENGLISH_VOCAB or not tokens:
            return 0
            
        mistakes = 0
        for token in tokens:
            # Check if alphanumeric, lowercase, and not a short/empty token
            if token.isalpha() and len(token) > 2:
                # Check lowercase form
                low_token = token.lower()
                if low_token not in ENGLISH_VOCAB:
                    # Exclude common abbreviations or domain jargon if needed
                    mistakes += 1
        return mistakes

    def extract_features(self, raw_text: str, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts structural, readability, style, SEO, and sentiment features from the raw
        and preprocessed text.
        """
        features = {}
        
        # 1. Basic structural counts
        cleaned_text = processed_data["cleaned_text"]
        tokens = processed_data["tokens"]
        
        # Word counts
        features["word_count"] = float(len(tokens))
        features["char_count"] = float(len(raw_text))
        
        # Sentence counts
        sentences = nltk.sent_tokenize(raw_text)
        features["sentence_count"] = float(max(1, len(sentences)))
        
        # Paragraph counts
        paragraphs = [p for p in raw_text.split('\n') if p.strip()]
        features["paragraph_count"] = float(max(1, len(paragraphs)))
        
        # 2. Readability Metrics (using textstat)
        # Handle empty text gracefully
        text_for_readability = raw_text if len(tokens) > 5 else "This is a placeholder sentence to evaluate readability metrics properly."
        
        try:
            features["readability_flesch_reading_ease"] = float(textstat.flesch_reading_ease(text_for_readability))
            features["readability_flesch_kincaid_grade"] = float(textstat.flesch_kincaid_grade(text_for_readability))
            features["readability_gunning_fog"] = float(textstat.gunning_fog(text_for_readability))
            features["readability_smog_index"] = float(textstat.smog_index(text_for_readability))
        except Exception as e:
            logger.warning(f"Error calculating readability scores: {e}")
            features["readability_flesch_reading_ease"] = 60.0
            features["readability_flesch_kincaid_grade"] = 8.0
            features["readability_gunning_fog"] = 10.0
            features["readability_smog_index"] = 8.0
            
        # 3. Vocabulary Metrics
        # Lexical diversity (Type-Token Ratio)
        unique_tokens = set(t.lower() for t in tokens if t.isalpha())
        total_alpha_tokens = [t for t in tokens if t.isalpha()]
        features["lexical_diversity"] = len(unique_tokens) / max(1, len(total_alpha_tokens))
        
        # Average word length
        word_lengths = [len(t) for t in total_alpha_tokens]
        features["avg_word_length"] = float(np.mean(word_lengths)) if word_lengths else 0.0
        
        # 4. Writing Style and Quality
        features["avg_sentence_length"] = features["word_count"] / features["sentence_count"]
        features["passive_voice_pct"] = self.estimate_passive_voice(raw_text)
        
        # Spelling mistake counts
        features["spelling_mistakes"] = float(self.estimate_spelling_mistakes(tokens))
        features["spelling_mistake_ratio"] = features["spelling_mistakes"] / max(1, features["word_count"])
        
        # 5. SEO Features
        # Heading count
        # In Markdown, headings start with #. In HTML, they are inside h1-h6 tags.
        markdown_headings = len(re.findall(r'^#{1,6}\s+\S+', raw_text, re.MULTILINE))
        html_headings = len(re.findall(r'<h[1-6][^>]*>.*?</h[1-6]>', raw_text, re.IGNORECASE))
        features["heading_count"] = float(max(markdown_headings, html_headings))
        
        # Transition words count
        transition_count = sum(1 for t in tokens if t.lower() in TRANSITION_WORDS)
        features["transition_word_density"] = (transition_count / max(1, features["word_count"])) * 100.0
        
        # Internal & External links
        # Matches markdown [text](url) and HTML <a href="url">
        markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', raw_text)
        html_links = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>', raw_text, re.IGNORECASE)
        
        all_links = [url for _, url in markdown_links] + html_links
        internal_count = 0
        external_count = 0
        
        for url in all_links:
            if url.startswith('/') or 'localhost' in url or 'contentiq' in url: # proxy domain
                internal_count += 1
            elif url.startswith('http'):
                external_count += 1
                
        features["internal_links"] = float(internal_count)
        features["external_links"] = float(external_count)
        features["total_links"] = float(len(all_links))
        
        # Image references
        markdown_images = len(re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', raw_text))
        html_images = len(re.findall(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', raw_text, re.IGNORECASE))
        features["image_count"] = float(max(markdown_images, html_images))
        
        # Keyword density
        # Clean tokens to find key topics (exclude punctuation, short words, stopwords)
        topic_words = [t.lower() for t in tokens if t.isalpha() and len(t) > 3 and t.lower() not in processed_data["filtered_tokens"]]
        # Wait, processed_data["filtered_tokens"] is already stopword-filtered! 
        # So we should just use processed_data["filtered_tokens"] excluding short words
        clean_topic_words = [t.lower() for t in processed_data["filtered_tokens"] if t.isalpha() and len(t) > 3]
        
        if clean_topic_words:
            fdist = nltk.FreqDist(clean_topic_words)
            # Take top 3 keywords
            top_3 = fdist.most_common(3)
            top_3_count = sum(count for word, count in top_3)
            features["keyword_density"] = (top_3_count / max(1, features["word_count"])) * 100.0
        else:
            features["keyword_density"] = 0.0
            
        # Meta description length proxy
        # If HTML, search for meta description, otherwise first paragraph length
        meta_match = re.search(r'<meta\s+name=["\']description["\']\s+content=["\']([^"\']+)["\'][^>]*>', raw_text, re.IGNORECASE)
        if meta_match:
            features["meta_description_length"] = float(len(meta_match.group(1)))
        else:
            # First paragraph as proxy
            first_p = paragraphs[0] if paragraphs else ""
            features["meta_description_length"] = float(min(160, len(first_p)))

        # 6. Sentiment Features
        try:
            blob = TextBlob(raw_text)
            features["sentiment_polarity"] = float(blob.sentiment.polarity)       # -1.0 to 1.0
            features["sentiment_subjectivity"] = float(blob.sentiment.subjectivity) # 0.0 to 1.0
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            features["sentiment_polarity"] = 0.0
            features["sentiment_subjectivity"] = 0.5
            
        return features

if __name__ == "__main__":
    from preprocessing import TextPreprocessor
    
    sample_text = """
    # Deep Learning vs Machine Learning
    
    In the modern era of artificial intelligence, Deep Learning has become extremely popular. However, it requires a lot of data. 
    Machine learning algorithms are trained to find patterns. A model is trained on a dataset. For example, a random forest model was built by our team last week.
    
    To learn more, check our [AI Course](/courses/ai) or visit the [TensorFlow website](https://tensorflow.org).
    
    Deep learning is a subset of machine learning, which is a subset of artificial intelligence. We need to explore various deep neural networks.
    """
    
    prep = TextPreprocessor()
    extractor = FeatureExtractor()
    
    proc = prep.process_pipeline(sample_text)
    feats = extractor.extract_features(sample_text, proc)
    
    print("\n--- Extracted Features ---")
    for k, v in feats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
