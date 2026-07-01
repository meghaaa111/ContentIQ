import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from config import logger
from predict import ContentPredictor
from recommendations import RecommendationEngine
import visualizations as vis

# Set Page Config
st.set_page_config(
    page_title="Results Dashboard - ContentIQ",
    page_icon="📊",
    layout="wide"
)

# Load Custom CSS
def load_css():
    css_path = Path(__file__).resolve().parent.parent / "assets" / "custom.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.sidebar.info("Use the navigation menu to analyze content or view the project details.")

# Initialize Session State variables if navigated to directly
if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None
if "history" not in st.session_state:
    st.session_state.history = []

# 1. Check for current analysis in session state. If empty, load default demo text.
if st.session_state.current_analysis is None:
    st.info("No active analysis found. Displaying default demo article analysis.")
    
    demo_text = """
    # Deep Learning vs Machine Learning Heuristics
    
    In the modern era of artificial intelligence, Deep Learning has become extremely popular. However, it requires a lot of data. 
    Machine learning algorithms are trained to find patterns. A model is trained on a dataset. For example, a random forest model was built by our team last week.
    
    To learn more, check our [AI Course](/courses/ai) or visit the [TensorFlow website](https://tensorflow.org).
    
    Deep learning is a subset of machine learning, which is a subset of artificial intelligence. We need to explore various deep neural networks.
    """
    try:
        predictor = ContentPredictor()
        demo_res = predictor.predict(demo_text)
        demo_res["recommendations"] = RecommendationEngine().generate_recommendations(demo_res["features"])
        st.session_state.current_analysis = demo_res
    except Exception as e:
        logger.error(f"Failed to generate demo analysis: {e}")
        st.error("Model artifacts are missing. Please complete training on the 'Model Performance' tab first.")
        st.stop()

# Retrieve active analysis
analysis = st.session_state.current_analysis

# 2. Top Metric Cards
badge_class = "badge-high" if analysis["prediction"] == "High Quality" else ("badge-medium" if analysis["prediction"] == "Medium Quality" else "badge-low")
status_color_class = "high" if analysis["prediction"] == "High Quality" else ("medium" if analysis["prediction"] == "Medium Quality" else "low")

st.markdown(
    f"""
    <div style="display: flex; gap: 15px; margin-bottom: 25px; flex-wrap: wrap;">
        <div class="glass-card score-card {status_color_class}" style="flex: 1; min-width: 200px; padding: 20px;">
            <div style="color: #9CA3AF; font-size: 0.85rem; font-weight: 500;">QUALITY VERDICT</div>
            <div class="score-value" style="margin: 10px 0;"><span class="badge {badge_class}" style="font-size:1.1rem; padding: 6px 12px;">{analysis['prediction']}</span></div>
        </div>
        <div class="glass-card score-card {status_color_class}" style="flex: 1; min-width: 200px; padding: 20px;">
            <div style="color: #9CA3AF; font-size: 0.85rem; font-weight: 500;">OVERALL QUALITY SCORE</div>
            <div class="score-value">{analysis['overall_score']:.1f}%</div>
        </div>
        <div class="glass-card score-card {status_color_class}" style="flex: 1; min-width: 200px; padding: 20px;">
            <div style="color: #9CA3AF; font-size: 0.85rem; font-weight: 500;">PREDICTION CONFIDENCE</div>
            <div class="score-value">{analysis['confidence'] * 100:.1f}%</div>
        </div>
        <div class="glass-card score-card" style="flex: 1; min-width: 200px; padding: 20px;">
            <div style="color: #9CA3AF; font-size: 0.85rem; font-weight: 500;">WORD COUNT</div>
            <div class="score-value">{int(analysis['features']['word_count'])}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# 3. Visualizations Layout Grid
tab1, tab2, tab3 = st.tabs(["Performance Gauges", "Vocabulary & Sentiment", "Explainable AI (SHAP)"])

with tab1:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(vis.plot_quality_gauge(analysis["overall_score"], analysis["prediction"]), use_container_width=True)
    with col_g2:
        st.markdown("<p style='text-align: center; color: #C0D5D6; font-size: 0.9rem; font-weight: 600; margin-bottom: 5px;'>Model Prediction Confidence</p>", unsafe_allow_html=True)
        st.plotly_chart(vis.plot_confidence_meter(analysis["confidence"]), use_container_width=True)
        
    st.markdown("---")
    col_g3, col_g4 = st.columns(2)
    with col_g3:
        st.plotly_chart(vis.plot_readability_comparison(analysis["features"]), use_container_width=True)
    with col_g4:
        st.plotly_chart(vis.plot_sentence_length_distribution(analysis["cleaned_text"]), use_container_width=True)

with tab2:
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.plotly_chart(vis.plot_sentiment_quadrant(analysis["features"]["sentiment_polarity"], analysis["features"]["sentiment_subjectivity"]), use_container_width=True)
    with col_v2:
        st.plotly_chart(vis.plot_word_frequencies(analysis["tokens"]), use_container_width=True)
        
    st.markdown("---")
    st.markdown("### Word Cloud Representation")
    wc_arr = vis.generate_word_cloud(analysis["cleaned_text"])
    st.image(wc_arr, use_column_width=True)

with tab3:
    st.markdown("### Model Interpretation via SHAP")
    st.markdown("<p style='font-size:0.9rem; color:#9CA3AF;'>Decision factors explain what specific document features pushed the machine learning model to make its quality prediction.</p>", unsafe_allow_html=True)
    
    col_sh1, col_sh2 = st.columns([3, 2])
    with col_sh1:
        if "all_stat_impacts" in analysis["shap_explanation"]:
            st.plotly_chart(vis.plot_local_shap_impacts(analysis["shap_explanation"]), use_container_width=True)
        else:
            st.warning("SHAP details not available.")
            
    with col_sh2:
        st.markdown("<h4 style='color:#A58D66; margin-top:0;'>Top Positive Factors (+ Quality)</h4>", unsafe_allow_html=True)
        if analysis["shap_explanation"]["positive"]:
            for factor in analysis["shap_explanation"]["positive"]:
                st.markdown(f"**{factor['feature']}** (impact: `+{factor['impact']:.3f}`)")
        else:
            st.markdown("<p style='font-size:0.875rem; color:#9CA3AF;'>None detected.</p>", unsafe_allow_html=True)
            
        st.markdown("<h4 style='color:#407E8C;'>Top Negative Factors (- Quality)</h4>", unsafe_allow_html=True)
        if analysis["shap_explanation"]["negative"]:
            for factor in analysis["shap_explanation"]["negative"]:
                st.markdown(f"**{factor['feature']}** (impact: `{factor['impact']:.3f}`)")
        else:
            st.markdown("<p style='font-size:0.875rem; color:#9CA3AF;'>None detected.</p>", unsafe_allow_html=True)

st.markdown("---")

# 4. Actionable Recommendations
st.markdown("### AI Recommendations Engine")
if not analysis["recommendations"]:
    st.success("ContentIQ found no major concerns. Your article aligns perfectly with SEO and readability guidelines!")
else:
    # Sort recommendations by priority: High, Medium, Low
    priority_order = {"High": 1, "Medium": 2, "Low": 3}
    sorted_recs = sorted(analysis["recommendations"], key=lambda x: priority_order.get(x["priority"], 9))
    
    # Render recommendations
    for r in sorted_recs:
        prio_class = "high-prio" if r["priority"] == "High" else ("med-prio" if r["priority"] == "Medium" else "low-prio")
        icon = f"[{r['priority']}]"
        
        st.markdown(
            f"""
            <div class="rec-item">
                <div class="rec-title {prio_class}">{icon} {r['priority']} Priority: {r['action']}</div>
                <div class="rec-body"><b>Issue:</b> {r['explanation']}</div>
                <div class="rec-impact"><b>SEO/User Impact:</b> {r['impact']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# 5. Exports & Downloads
st.markdown("### Export Audit Reports")
col_exp1, col_exp2 = st.columns(2)

# Prepare CSV feature table
features_df = pd.DataFrame(list(analysis["features"].items()), columns=["Metric Feature", "Calculated Value"])
csv_data = features_df.to_csv(index=False).encode('utf-8')

# Prepare Markdown Audit Report
md_report = f"""# ContentIQ Quality Audit Report

**Verdict class:** {analysis['prediction']}
**Overall Quality Score:** {analysis['overall_score']:.1f}%
**Confidence:** {analysis['confidence']*100:.1f}%
**Word Count:** {int(analysis['features']['word_count'])}

---

## Metric Breakdown
"""
for k, v in analysis["features"].items():
    md_report += f"- **{k.replace('_', ' ').title()}:** {v:.4f}\n"

md_report += "\n---\n\n## Recommendations\n"
for idx, r in enumerate(analysis["recommendations"]):
    md_report += f"### {idx+1}. [{r['priority']} Priority] {r['action']}\n- **Explanation:** {r['explanation']}\n- **Impact:** {r['impact']}\n\n"

with col_exp1:
    st.download_button(
        label="Download Metrics CSV Table",
        data=csv_data,
        file_name="contentiq_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )
    
with col_exp2:
    st.download_button(
        label="Export Markdown Audit Report",
        data=md_report.encode("utf-8"),
        file_name="contentiq_audit_report.md",
        mime="text/markdown",
        use_container_width=True
    )
