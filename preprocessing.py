import re
import string
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from config import logger

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            if resource == 'punkt':
                nltk.data.find("tokenizers/punkt")
            elif resource == 'punkt_tab':
                nltk.data.find("tokenizers/punkt_tab")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            logger.info(f"NLTK resource '{resource}' not found. Downloading...")
            nltk.download(resource, quiet=True)

download_nltk_resources()

class TextPreprocessor:
    """
    Modular text preprocessing pipeline for NLP feature extraction and ML modeling.
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Add basic punctuation to stopwords if needed, but we do that separately
        
    def remove_html(self, text: str) -> str:
        """Removes HTML tags from the text."""
        if not text:
            return ""
        # Use BeautifulSoup to parse and extract raw text
        try:
            soup = BeautifulSoup(text, "html.parser")
            return soup.get_text()
        except Exception as e:
            logger.warning(f"BeautifulSoup parsing failed, falling back to regex: {e}")
            return re.sub(r'<[^>]*>', '', text)

    def remove_urls(self, text: str) -> str:
        """Removes URLs and hyperlinks."""
        # Pattern matches http/https and www links
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_emojis(self, text: str) -> str:
        """Removes emojis and unicode symbols."""
        # Replace non-ASCII/standard characters and unicode symbols
        # Matches emoji range and miscellaneous symbols
        emoji_pattern = re.compile(
            '['
            '\U00010000-\U0010ffff'  # Emoji ranges
            '\u2600-\u27BF'          # Misc symbols & Dingbats
            '\u2000-\u32FF'          # General punctuation, CJK, etc.
            ']+', 
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def clean_punctuation_and_special(self, text: str) -> str:
        """Removes standard punctuation and special characters, leaving clean spacing."""
        # Replace punctuation with a space to prevent joining words
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
        cleaned = text.translate(translator)
        # Collapse multiple spaces into one
        return re.sub(r'\s+', ' ', cleaned).strip()

    def clean_text(self, text: str) -> str:
        """
        Combines modular text cleaning steps: lowercasing, HTML removal,
        URL removal, emoji removal, and character cleaning.
        """
        if not text:
            return ""
        
        # 1. Lowercase
        cleaned = text.lower()
        
        # 2. Remove HTML
        cleaned = self.remove_html(cleaned)
        
        # 3. Remove URLs
        cleaned = self.remove_urls(cleaned)
        
        # 4. Remove Emojis
        cleaned = self.remove_emojis(cleaned)
        
        # 5. Clean punctuation and multiple spaces
        cleaned = self.clean_punctuation_and_special(cleaned)
        
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Splits text into token words."""
        if not text:
            return []
        return nltk.word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Removes standard English stopwords from token list."""
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatizes tokens to their base form."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def process_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Executes the full pipeline sequentially and returns detailed stages.
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize(filtered_tokens)
        
        return {
            "raw_text": text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "processed_text": " ".join(lemmatized_tokens)
        }

if __name__ == "__main__":
    # Quick self-test
    sample = "<html><body><h1>Hello World! 😊</h1> Check out https://google.com for more. Coding is running...</body></html>"
    preprocessor = TextPreprocessor()
    res = preprocessor.process_pipeline(sample)
    print("Raw text:", res["raw_text"])
    print("Cleaned text:", res["cleaned_text"])
    print("Lemmatized:", res["lemmatized_tokens"])
