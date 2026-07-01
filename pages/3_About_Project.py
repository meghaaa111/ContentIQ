import streamlit as st
from pathlib import Path

# Set Page Config
st.set_page_config(
    page_title="About Project - ContentIQ",
    layout="wide"
)

# Load Custom CSS
def load_css():
    css_path = Path(__file__).resolve().parent.parent / "assets" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize Session State variables if navigated to directly
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.info("Use the navigation menu to analyze content or view the project details.")

st.markdown("<h1 class='hero-title'>About ContentIQ</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Production-grade AI pipeline specifications and NLP architectures.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="margin-top:0; color:#A58D66;">Pipeline Architecture</h3>
            <p style="color:#E5E1DD; line-height:1.6; font-size:0.95rem;">
                ContentIQ functions as an automated pre-publishing auditor. The pipeline transforms raw text 
                into scaled feature matrices, evaluates the metrics via ensemble machine learning models, 
                and extracts positive/negative decision factors.
            </p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin: 20px 0; font-weight: 600; font-size: 0.75rem; justify-content: center; align-items: center; color: #C0D5D6; background: rgba(5, 37, 51, 0.4); padding: 12px; border-radius: 8px; border: 1px solid rgba(64, 126, 148, 0.15);">
                <span>EXTRACTION</span>
                <span style="color:#A58D66;">➔</span>
                <span>CLEANING</span>
                <span style="color:#A58D66;">➔</span>
                <span>METRICS MATH</span>
                <span style="color:#A58D66;">➔</span>
                <span>TF-IDF</span>
                <span style="color:#A58D66;">➔</span>
                <span>CLASSIFIER</span>
                <span style="color:#A58D66;">➔</span>
                <span>SHAP EXPLANATION</span>
                <span style="color:#A58D66;">➔</span>
                <span>RECOMMENDATIONS</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Detailed Steps in collapsed expander to keep the page clean
    with st.expander("Detailed Pipeline Processing Steps Breakdown"):
        st.markdown(
            """
            1. **Input Extraction:** Parses and reads uploaded text documents (`.txt`, `.docx`, `.pdf`) into clean string characters.
            2. **Modular Preprocessing:** Strips HTML/markup formatting tags, links, and emojis, followed by NLTK word tokenization.
            3. **Feature Engineering:** Extracts counts, readability scores (Flesch, Kincaid, SMOG, Fog), Type-Token Ratios, passive voice verbs, and sentiment polarity/subjectivity.
            4. **Semantic Vectorization:** Fits a 50-dimensional TF-IDF array to capture topic semantic n-grams.
            5. **Tabular Alignment:** Scales statistical features using a standard scaler and joins them with the sparse TF-IDF array.
            6. **Classifier Scoring:** Computes predictions using Logistic Regression, Random Forest, or XGBoost.
            7. **Local Interpretation:** Computes SHAP decision weights or linear coefficients to explain features contributing to the quality label.
            8. **Actionable Recommendations:** Runs rule-based conditional flags to detail improvement priorities (High/Medium/Low).
            """
        )

with col2:
    st.markdown(
        """
        <div class="glass-card" style="border-left: 5px solid #407E8C;">
            <h3 style="margin-top:0; color:#407E8C; margin-bottom:15px;">Technology Specifications</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Render neat tech specifications table
    st.markdown(
        """
        | Pipeline Phase | Libraries Used | Specific Role |
        | :--- | :--- | :--- |
        | **NLP Core** | `NLTK`, `BeautifulSoup4` | Tokenization, lemmatization, and HTML cleaning |
        | **Readability** | `Textstat` | Readability grade indexes calculation (Flesch, Fog, SMOG) |
        | **Style / Sentiment** | `TextBlob` | Polarity, subjectivity, spelling estimations |
        | **Classification** | `Scikit-Learn`, `XGBoost` | Feature scaling, TF-IDF vectorization, model scoring |
        | **Explainable AI** | `SHAP` | SHapley Additive exPlanations feature weights |
        | **Document Parsers** | `PyPDF`, `python-docx` | PDF and MS Word document reading |
        """
    )
    
    st.markdown(
        """
        <div class="glass-card" style="border-left: 5px solid #A58D66; margin-top:20px;">
            <h3 style="margin-top:0; color:#A58D66;">SEO Relevance</h3>
            <p style="color:#E5E1DD; font-size:0.875rem; line-height:1.5; margin-bottom:0;">
                ContentIQ emulates modern search quality parameters (like E-E-A-T: Experience, Expertise, Authoritativeness, and Trustworthiness), 
                identifying keyword stuffing, AI-like phrasing, and poor layout structures before crawler indexing.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
