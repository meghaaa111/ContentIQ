import streamlit as st
from pathlib import Path
from config import logger

# Set page config
st.set_page_config(
    page_title="ContentIQ - Content Quality Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None

# Load Custom CSS
def load_css():
    css_path = Path(__file__).resolve().parent / "assets" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        logger.warning("Custom CSS file not found.")

load_css()

st.sidebar.info("Use the navigation menu to analyze content or view the project details.")

# Main Page Layout
st.markdown("<h1 class='hero-title'>Content Quality Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>Evaluate your written content across SEO, readability, style, and tone using machine learning and natural language processing.</p>", unsafe_allow_html=True)

# Main Dashboard Container (Glassmorphism layout)
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        <div class="glass-card">
            <h3 style='margin-top:0; color:#A58D66;'>Evaluate Pre-Publishing Risks</h3>
            <p style='color:#E5E1DD; line-height:1.6;'>
                ContentIQ emulates how search engine crawlers and professional human editors evaluate content before publication.
                Instead of simple spelling checkers, our pipeline analyzes the semantic quality, readability grade levels, link health, 
                and vocabulary richness to protect your search rankings and build trust with readers.
            </p>
            <h4 style='color:#C0D5D6;'>Core Analysis Dimensions:</h4>
            <ul style='color:#E5E1DD; line-height:1.8; margin-left:20px;'>
                <li><b>SEO Optimization:</b> Computes heading frequency, keyword density, transition patterns, and internal/external hyperlink distributions.</li>
                <li><b>Readability Formulas:</b> Calculates Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog, and SMOG grades.</li>
                <li><b>Style & Grammar Heuristics:</b> Measures sentence-length variance, active vs. passive voice ratios, and spelling error rates.</li>
                <li><b>Vocabulary & Linguistics:</b> Audits lexical diversity (Type-Token Ratios) to guarantee rich, non-repetitive prose.</li>
                <li><b>Sentiment & Tone:</b> Computes semantic polarity and subjectivity levels to align tone with brand standards.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("### How It Works")
    step_col1, step_col2, step_col3 = st.columns(3)
    
    with step_col1:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; min-height: 180px;">
                <h4 style="color:#A58D66; margin-top:0;">1. Input Content</h4>
                <p style="font-size:0.875rem; color:#C0D5D6;">Paste an article or upload TXT, Word, or PDF documents directly.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with step_col2:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; min-height: 180px;">
                <h4 style="color:#407E8C; margin-top:0;">2. ML Quality Score</h4>
                <p style="font-size:0.875rem; color:#C0D5D6;">Tabular statistics and TF-IDF features are analyzed by trained XGBoost & Random Forest models.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with step_col3:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; min-height: 180px;">
                <h4 style="color:#C0D5D6; margin-top:0;">3. Optimize & Export</h4>
                <p style="font-size:0.875rem; color:#C0D5D6;">Receive SHAP explainability insights, detailed recommendations, and download reports.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

with col2:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <h3 style="margin-top:0; color:#A58D66;">Get Started</h3>
            <p style="font-size:0.9rem; color:#E5E1DD; margin-bottom:20px;">Ready to audit your blog post, copy, or essay?</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <a href="/Analyze_Content" target="_self" style="text-decoration: none; color: inherit;">
            <div class="nav-button">
                Analyze New Content
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
        
    st.markdown("---")
    st.markdown("### Recent History")
    
    if not st.session_state.history:
        st.markdown(
            """
            <div style="background: rgba(255,255,255,0.03); border: 1px dashed rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; text-align: center; color: #C0D5D6;">
                No articles analyzed in this session yet.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for idx, hist in enumerate(st.session_state.history[:3]):
            badge_class = "badge-high" if hist["prediction"] == "High Quality" else ("badge-medium" if hist["prediction"] == "Medium Quality" else "badge-low")
            st.markdown(
                f"""
                <div class="glass-card" style="padding:12px; margin-bottom:10px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="font-size:0.9rem; color:#FFFFFF; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:180px;">{hist['title']}</b>
                        <span class="badge {badge_class}">{hist['prediction']}</span>
                    </div>
                    <div style="font-size:0.8rem; color:#C0D5D6; margin-top:5px;">Score: {hist['score']:.1f}% | Words: {int(hist['words'])}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Simple link to load in results
            if st.button(f"Load Result #{idx+1}", key=f"hist_load_{idx}"):
                st.session_state.current_analysis = hist["result_data"]
                st.success("Result loaded! [Go to Results Dashboard](/Results_Dashboard)")
