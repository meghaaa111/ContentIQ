import streamlit as st
import docx
from pypdf import PdfReader
from pathlib import Path

from config import logger
from predict import ContentPredictor
from recommendations import RecommendationEngine

# Set Page Config
st.set_page_config(
    page_title="Analyze Content - ContentIQ",
    page_icon="📝",
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

st.markdown("<h1 class='hero-title'>Content Quality Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Paste your text, upload documents, and customize preprocessing configurations.</p>", unsafe_allow_html=True)

# Helper function to read file contents
def extract_text_from_file(uploaded_file) -> str:
    file_name = uploaded_file.name
    extracted_text = ""
    
    try:
        if file_name.endswith('.txt'):
            extracted_text = uploaded_file.read().decode("utf-8")
        elif file_name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            extracted_text = "\n".join([p.text for p in doc.paragraphs])
        elif file_name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            st.error("Unsupported file format.")
    except Exception as e:
        logger.error(f"Error parsing file: {e}", exc_info=True)
        st.error(f"Failed to parse file: {e}")
        
    return extracted_text

# Layout columns for inputs and options
col_input, col_options = st.columns([2, 1])

# Initialize text
input_text = ""
document_title = "Pasted Text"

with col_input:
    st.markdown("### Enter Content")
    
    # Selection tabs
    input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"], horizontal=True)
    
    if input_method == "Paste Text":
        input_text = st.text_area(
            "Copy and paste your draft here (minimum 30 words recommended):",
            height=350,
            placeholder="Type or paste your article content here..."
        )
    else:
        uploaded_file = st.file_uploader("Upload an article (TXT, DOCX, or PDF):", type=["txt", "docx", "pdf"])
        if uploaded_file is not None:
            document_title = uploaded_file.name
            input_text = extract_text_from_file(uploaded_file)
            
            # Show preview
            with st.expander(f"Uploaded File Preview ({document_title})", expanded=True):
                st.text_area("File content:", input_text[:1200] + ("\n... [Truncated]" if len(input_text) > 1200 else ""), height=180, disabled=True)

with col_options:
    st.markdown("### Preprocessing Options")
    st.markdown("<p style='font-size:0.875rem; color:#9CA3AF;'>Configure the content cleaning parameters prior to classification.</p>", unsafe_allow_html=True)
    
    clean_html = st.checkbox("Strip HTML Tags", value=True, help="Removes paragraph, formatting, and structural tags from content.")
    clean_urls = st.checkbox("Remove URLs & Hyperlinks", value=True, help="Replaces links with spaces to prevent skewed length features.")
    clean_emojis = st.checkbox("Strip Emojis & Symbols", value=True, help="Removes non-ascii unicode emojis.")
    
    st.markdown("---")
    st.markdown("### Execute Quality Check")
    
    run_analysis = st.button("Run ContentIQ Assessment", use_container_width=True)

# Process logic
if run_analysis:
    if not input_text.strip():
        st.warning("Please enter or upload some content before starting analysis.")
    elif len(input_text.split()) < 10:
        st.warning("Content is too short (minimum 10 words required). Please expand your text.")
    else:
        with st.spinner("Analyzing content properties, calculating SHAP weights, and formatting recommendations..."):
            try:
                # Load predictor and predict
                predictor = ContentPredictor()
                result = predictor.predict(input_text)
                
                # Generate recommendations
                engine = RecommendationEngine()
                recs = engine.generate_recommendations(result["features"])
                result["recommendations"] = recs
                
                # Save current results in session state
                st.session_state.current_analysis = result
                
                # Add to session history
                word_cnt = result["features"]["word_count"]
                
                history_entry = {
                    "title": document_title if len(document_title) < 25 else document_title[:22] + "...",
                    "prediction": result["prediction"],
                    "score": result["overall_score"],
                    "words": word_cnt,
                    "result_data": result
                }
                
                # Insert at the beginning of history list
                st.session_state.history.insert(0, history_entry)
                
                st.success("ContentIQ Analysis Complete!")
                
                # Dynamic info card
                badge_class = "badge-high" if result["prediction"] == "High Quality" else ("badge-medium" if result["prediction"] == "Medium Quality" else "badge-low")
                
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:16px;">
                        <h4 style="margin-top:0;">Assessment Summary</h4>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span style="color:#D1D5DB;">Label Verdict:</span>
                            <span class="badge {badge_class}">{result['prediction']}</span>
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="color:#D1D5DB;">Overall Score:</span>
                            <b style="color:#60A5FA; font-size:1.15rem;">{result['overall_score']:.1f}%</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.info("Navigate to the **Results Dashboard** in the left sidebar to view interactive charts, explainability metrics, and actionable recommendations.")
                
            except Exception as e:
                logger.error(f"Error running inference: {e}", exc_info=True)
                st.error(f"Inference pipeline encountered an error: {e}. Please ensure the ML model is trained by running 'train.py'.")
