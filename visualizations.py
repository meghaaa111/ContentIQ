import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import random
from typing import List, Dict, Any, Tuple
import nltk

# Palette Constants
COLOR_NAVY = "#083A4F"
COLOR_GOLD = "#A58D66"
COLOR_AQUA = "#C0D5D6"
COLOR_TEAL = "#407E8C"
COLOR_SAND = "#E5E1DD"

def plot_quality_gauge(score: float, label: str) -> go.Figure:
    """Generates a gauge chart for the overall content quality score styled after the Navy-Gold-Teal palette."""
    if label == "High Quality":
        color = COLOR_GOLD
    elif label == "Medium Quality":
        color = COLOR_TEAL
    else:
        color = COLOR_AQUA

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=score,
        title={'text': f"Overall Quality: {label}", 'font': {'size': 15, 'color': COLOR_SAND}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': COLOR_AQUA},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "rgba(165, 141, 102, 0.3)", # Gold border
            'steps': [
                {'range': [0, 40], 'color': 'rgba(192, 213, 214, 0.08)'},  # Aqua step
                {'range': [40, 75], 'color': 'rgba(64, 126, 148, 0.12)'}, # Teal step
                {'range': [75, 100], 'color': 'rgba(165, 141, 102, 0.12)'} # Gold step
            ],
            'threshold': {
                'line': {'color': COLOR_SAND, 'width': 3},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    # Place a clean, centered text annotation inside the hollow arc
    fig.add_annotation(
        x=0.5,
        y=0.22,
        text=f"{score:.1f}%",
        showarrow=False,
        font=dict(color=color, size=38, family="Inter", weight="bold")
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SAND, 'family': "Inter, sans-serif"},
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def plot_confidence_meter(confidence: float) -> go.Figure:
    """Generates a progress indicator for prediction confidence using Teal and Gold highlights."""
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=confidence * 100,
        number={'suffix': "%", 'font': {'color': COLOR_GOLD, 'size': 32}},
        gauge={
            'shape': 'bullet',
            'axis': {'range': [0, 100]},
            'bar': {'color': COLOR_TEAL},
            'bgcolor': "rgba(8, 58, 79, 0.5)", # Navy bg
            'steps': [
                {'range': [0, 50], 'color': 'rgba(192, 213, 214, 0.08)'},
                {'range': [50, 100], 'color': 'rgba(165, 141, 102, 0.08)'}
            ]
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=100,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    return fig

def plot_sentiment_quadrant(polarity: float, subjectivity: float) -> go.Figure:
    """Generates a scatter plot mapping the sentiment polarity and subjectivity using custom quadrants."""
    fig = go.Figure()
    
    # Background layout quadrant markers using soft steps of Aqua and Gold
    fig.add_shape(type="rect", x0=-1.0, y0=0.0, x1=0.0, y1=0.5, line_width=0, fillcolor="rgba(192, 213, 214, 0.03)") # Neg-Objective (Aqua)
    fig.add_shape(type="rect", x0=-1.0, y0=0.5, x1=0.0, y1=1.0, line_width=0, fillcolor="rgba(192, 213, 214, 0.07)")  # Neg-Subjective
    fig.add_shape(type="rect", x0=0.0, y0=0.0, x1=1.0, y1=0.5, line_width=0, fillcolor="rgba(165, 141, 102, 0.03)") # Pos-Objective (Gold)
    fig.add_shape(type="rect", x0=0.0, y0=0.5, x1=1.0, y1=1.0, line_width=0, fillcolor="rgba(165, 141, 102, 0.07)")  # Pos-Subjective
    
    # Add coordinate crosshairs in Teal
    fig.add_shape(type="line", x0=-1.0, y0=0.5, x1=1.0, y1=0.5, line=dict(color=COLOR_TEAL, width=1, dash="dash"))
    fig.add_shape(type="line", x0=0.0, y0=0.0, x1=0.0, y1=1.0, line=dict(color=COLOR_TEAL, width=1, dash="dash"))
    
    # Plot user score point using Gold marker with white ring
    fig.add_trace(go.Scatter(
        x=[polarity],
        y=[subjectivity],
        mode='markers+text',
        marker=dict(size=16, color=COLOR_GOLD, line=dict(color=COLOR_SAND, width=2)),
        text=["Your Article"],
        textposition="top center",
        textfont=dict(color=COLOR_SAND, size=12, family="Inter")
    ))
    
    fig.update_layout(
        title={'text': "Sentiment Tone Quadrant", 'font': {'size': 16, 'color': COLOR_SAND}},
        xaxis=dict(title=dict(text="Polarity (Negative ↔ Positive)", font=dict(color=COLOR_AQUA)), range=[-1.0, 1.0], gridcolor='rgba(192, 213, 214, 0.1)', tickcolor=COLOR_SAND),
        yaxis=dict(title=dict(text="Subjectivity (Objective ↔ Subjective)", font=dict(color=COLOR_AQUA)), range=[0.0, 1.0], gridcolor='rgba(192, 213, 214, 0.1)', tickcolor=COLOR_SAND),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SAND, 'family': "Inter, sans-serif"},
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_local_shap_impacts(shap_explanation: Dict[str, Any]) -> go.Figure:
    """Generates a horizontal bar chart displaying SHAP feature impacts matching custom palette."""
    pos_factors = shap_explanation.get("positive", [])
    neg_factors = shap_explanation.get("negative", [])
    
    all_factors = pos_factors + neg_factors
    if not all_factors:
        fig = go.Figure()
        fig.update_layout(
            title="SHAP Feature Importance (No Data)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': COLOR_SAND}
        )
        return fig
        
    df = pd.DataFrame(all_factors)
    df = df.sort_values(by="impact", ascending=True)
    
    # Gold for positive impact (improves quality), Teal for negative (reduces quality)
    colors = [COLOR_TEAL if x < 0 else COLOR_GOLD for x in df['impact']]
    
    fig = go.Figure(go.Bar(
        x=df['impact'],
        y=df['feature'],
        orientation='h',
        marker_color=colors,
        hovertemplate="Feature: %{y}<br>SHAP Impact: %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title={'text': "Top Quality Decision Factors (Explainable SHAP)", 'font': {'size': 16, 'color': COLOR_SAND}},
        xaxis=dict(title=dict(text="SHAP Impact (← Reduces Quality  |  Increases Quality →)", font=dict(color=COLOR_AQUA)), gridcolor='rgba(192, 213, 214, 0.15)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SAND, 'family': "Inter, sans-serif"},
        height=320,
        margin=dict(l=150, r=30, t=50, b=50)
    )
    return fig

def plot_word_frequencies(tokens: List[str]) -> go.Figure:
    """Generates a horizontal bar chart showing the frequency of the top keywords."""
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    
    words_list = [w.lower() for w in tokens if w.isalpha() and len(w) > 2 and w.lower() not in stop_words]
    
    if not words_list:
        fig = go.Figure()
        fig.update_layout(title="Top Keywords Frequency (No Data)", paper_bgcolor='rgba(0,0,0,0)')
        return fig
        
    fdist = nltk.FreqDist(words_list)
    top_10 = fdist.most_common(10)
    
    df = pd.DataFrame(top_10, columns=["Keyword", "Frequency"])
    df = df.sort_values(by="Frequency", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["Frequency"],
        y=df["Keyword"],
        orientation='h',
        marker_color=COLOR_TEAL, # Teal bars
        hovertemplate="Keyword: %{y}<br>Frequency: %{x}<extra></extra>"
    ))
    
    fig.update_layout(
        title={'text': "Top Topic Keywords Frequency", 'font': {'size': 16, 'color': COLOR_SAND}},
        xaxis=dict(title=dict(text="Occurrences", font=dict(color=COLOR_AQUA)), gridcolor='rgba(192, 213, 214, 0.15)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SAND, 'family': "Inter, sans-serif"},
        height=320,
        margin=dict(l=100, r=20, t=50, b=40)
    )
    return fig

def plot_sentence_length_distribution(raw_text: str) -> go.Figure:
    """Generates a histogram showing the distribution of sentence lengths."""
    sentences = nltk.sent_tokenize(raw_text)
    lengths = [len(nltk.word_tokenize(s)) for s in sentences if s.strip()]
    
    if not lengths:
        lengths = [0]
        
    df = pd.DataFrame({"Word Count": lengths})
    avg_len = np.mean(lengths)
    
    fig = go.Figure()
    
    # Histogram in soft transparent Aqua
    fig.add_trace(go.Histogram(
        x=df["Word Count"],
        xbins=dict(size=4),
        marker_color=COLOR_AQUA,
        opacity=0.75,
        hovertemplate="Sentence length bin: %{x}<br>Count: %{y}<extra></extra>"
    ))
    
    # Average vertical line in Gold
    fig.add_shape(
        type="line",
        x0=avg_len, y0=0, x1=avg_len, y1=1,
        yref="paper",
        line=dict(color=COLOR_GOLD, width=2, dash="dash")
    )
    
    fig.add_annotation(
        x=avg_len, y=0.9,
        yref="paper",
        text=f"Average: {avg_len:.1f} words",
        showarrow=True,
        arrowhead=1,
        ax=40, ay=-30,
        font=dict(color=COLOR_GOLD, size=11),
        arrowcolor=COLOR_GOLD
    )
    
    fig.update_layout(
        title={'text': "Sentence Length Distribution", 'font': {'size': 16, 'color': COLOR_SAND}},
        xaxis=dict(title=dict(text="Words per Sentence", font=dict(color=COLOR_AQUA)), gridcolor='rgba(192, 213, 214, 0.15)'),
        yaxis=dict(title=dict(text="Sentence Count", font=dict(color=COLOR_AQUA)), gridcolor='rgba(192, 213, 214, 0.15)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SAND, 'family': "Inter, sans-serif"},
        height=320,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig

def plot_readability_comparison(features: Dict[str, float]) -> go.Figure:
    """Generates a bar chart comparing readability indices against optimal guidelines."""
    metrics = {
        "Flesch-Kincaid Grade": features.get("readability_flesch_kincaid_grade", 8.0),
        "Gunning Fog Index": features.get("readability_gunning_fog", 10.0),
        "SMOG Index": features.get("readability_smog_index", 8.0)
    }
    
    df = pd.DataFrame(list(metrics.items()), columns=["Index", "Grade Level"])
    
    fig = go.Figure()
    
    # Target reference zone in light Gold
    fig.add_shape(
        type="rect",
        x0=-0.5, y0=6.0, x1=2.5, y1=10.0,
        line_width=0,
        fillcolor="rgba(165, 141, 102, 0.12)" # Light gold
    )
    
    # Gold bars for target reading levels, Teal for outliers
    colors = [COLOR_GOLD if 6.0 <= val <= 10.0 else COLOR_TEAL for val in df["Grade Level"]]
    
    fig.add_trace(go.Bar(
        x=df["Index"],
        y=df["Grade Level"],
        marker_color=colors,
        hovertemplate="%{x}: Grade %{y:.1f}<extra></extra>"
    ))
    
    fig.add_annotation(
        x=1.0, y=8.0,
        text="Ideal Range (Grades 6-10)",
        showarrow=False,
        font=dict(color=COLOR_GOLD, size=12, family="Inter")
    )
    
    fig.update_layout(
        title={'text': "Readability Grade Comparison", 'font': {'size': 16, 'color': COLOR_SAND}},
        xaxis=dict(gridcolor='rgba(0,0,0,0)'),
        yaxis=dict(title=dict(text="Target Grade Level (Lower is easier)", font=dict(color=COLOR_AQUA)), gridcolor='rgba(192, 213, 214, 0.15)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLOR_SAND, 'family': "Inter, sans-serif"},
        height=320,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig

def generate_word_cloud(cleaned_text: str) -> np.ndarray:
    """Generates a styled word cloud matching the Navy, Gold, Aqua, Teal, Sand theme."""
    if not cleaned_text.strip():
        cleaned_text = "ContentIQ quality writing analysis SEO dashboard"
        
    # Custom color function generating random theme colors
    def palette_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        theme_colors = [COLOR_GOLD, COLOR_AQUA, COLOR_TEAL, COLOR_SAND]
        return random.choice(theme_colors)
        
    try:
        # Generates a PNG with transparent background
        wc = WordCloud(
            width=800, height=400,
            background_color="rgba(0,0,0,0)",
            mode="RGBA",
            max_words=100,
            color_func=palette_color_func
        ).generate(cleaned_text)
        return wc.to_array()
    except Exception as e:
        return np.zeros((400, 800, 4), dtype=np.uint8)
