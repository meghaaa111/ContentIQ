<div align="center">

<img src="assets/banner.svg" alt="ContentIQ Logo" width="90" />

# ContentIQ

### AI-Powered Content Intelligence & Pre-Publishing Audit Platform

Evaluate written content before it goes live — quality, readability, SEO, and style, scored and explained in one place.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-083A4F?style=for-the-badge&logo=python&logoColor=C0D5D6)
![Streamlit](https://img.shields.io/badge/Streamlit-App-A58D66?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-407E8C?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-083A4F?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-A58D66?style=for-the-badge&logo=plotly&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-407E8C?style=for-the-badge)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-083A4F?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Text_Processing-A58D66?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-C0D5D6?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/username/ContentIQ?style=for-the-badge&color=407E8C)
![Stars](https://img.shields.io/github/stars/username/ContentIQ?style=for-the-badge&color=A58D66)
![Forks](https://img.shields.io/github/forks/username/ContentIQ?style=for-the-badge&color=083A4F)

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Streamlit_App-407E8C?style=for-the-badge)](https://your-streamlit-link.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/📂_GitHub_Repository-083A4F?style=for-the-badge)](https://github.com/username/ContentIQ)

</div>

<br/>

## 📖 About ContentIQ

ContentIQ is a pre-publishing content audit platform that scores writing quality before it reaches an audience. It exists because most writers only discover weak readability, thin SEO, or poor structure after publishing — when the cost of fixing it is highest. ContentIQ closes that gap by combining NLP, feature engineering, and machine learning into a single audit pass. It benefits bloggers, content teams, SEO specialists, and technical writers who want editor-level feedback without waiting on a human editor. Every prediction is explained, not just scored, so users know exactly what to fix and why.

<br/>

## ✨ Key Features

- ✅ **Content Quality Classification** — predicts an overall quality label using a trained ML model.
- ✅ **NLP-Based Text Processing** — cleans, tokenizes, and structures raw text for analysis.
- ✅ **SEO Content Analysis** — evaluates keyword usage, density, and on-page SEO signals.
- ✅ **Readability Assessment** — scores content using standard readability formulas.
- ✅ **Vocabulary Richness Analysis** — measures lexical diversity and word repetition.
- ✅ **Stylistic Heuristic Evaluation** — flags tone, sentence variety, and structural patterns.
- ✅ **Explainable AI (SHAP)** — reveals which features drove each prediction.
- ✅ **AI Recommendation Engine** — converts weak metrics into prioritized action items.
- ✅ **Interactive Analytics Dashboard** — visualizes scores and feature breakdowns in real time.
- ✅ **Report Export (CSV & Markdown)** — download audit results for offline use.

<br/>

## ⚙️ Technical Pipeline

```mermaid
flowchart TD
    A[Text / Document Input] --> B[Document Extraction]
    B --> C[Preprocessing]
    C --> D[Feature Engineering]
    D --> E[TF-IDF Embeddings]
    E --> F[Machine Learning Prediction]
    F --> G[SHAP Explainability]
    G --> H[Recommendation Engine]
    H --> I[Interactive Dashboard]
    I --> J[Report Export]

    style A fill:#083A4F,color:#fff
    style B fill:#407E8C,color:#fff
    style C fill:#407E8C,color:#fff
    style D fill:#A58D66,color:#fff
    style E fill:#A58D66,color:#fff
    style F fill:#083A4F,color:#fff
    style G fill:#083A4F,color:#fff
    style H fill:#407E8C,color:#fff
    style I fill:#A58D66,color:#fff
    style J fill:#083A4F,color:#fff
```

<br/>

## 🧠 Feature Engineering

| Category | Examples | Purpose |
|---|---|---|
| **SEO** | Keyword density, meta signals, heading usage | Measure search-readiness |
| **Readability** | Flesch-Kincaid, sentence length, syllable count | Gauge ease of reading |
| **Vocabulary** | Type-token ratio, unique word count | Measure lexical richness |
| **Style** | Passive voice, sentence variety, tone markers | Evaluate writing craft |
| **TF-IDF Embeddings** | Weighted term vectors | Capture textual signal for ML |

<br/>

## 🤖 Machine Learning Pipeline

- **Dataset Generation** — labeled content samples built from extracted feature sets.
- **Model Training** — supervised training across candidate classifiers.
- **Model Selection** — best model chosen via cross-validated performance.
- **Prediction** — quality label generated for new, unseen content.
- **Confidence Score** — probability estimate attached to every prediction.
- **SHAP Explainability** — per-prediction feature attribution for transparency.

<br/>

## 📊 Explainable AI

ContentIQ uses SHAP to break down every prediction into the individual features that influenced it. Instead of a black-box score, users see exactly which readability, SEO, or style signals pushed the result up or down. This turns each audit into an actionable, interpretable report rather than a single number.

<br/>

## 💡 AI Recommendation Engine

Recommendations are generated by comparing extracted metrics against ideal publishing thresholds. Each gap is translated into a clear, actionable suggestion and tagged by urgency:

- 🔴 **High Priority** — issues likely to hurt quality or SEO significantly.
- 🟡 **Medium Priority** — improvements worth addressing before publishing.
- 🟢 **Low Priority** — minor polish suggestions.

<br/>

## 📈 Screenshots

<div align="center">

| Home | Analyze Content |
|---|---|
| ![Home](assets/home.png) | ![Analyze Content](assets/analyze.png) |

| Results Dashboard | Recommendations |
|---|---|
| ![Results Dashboard](assets/results.png) | ![Recommendations](assets/recommendations.png) |

| Model Performance | About Project |
|---|---|
| ![Model Performance](assets/model_performance.png) | ![About Project](assets/about.png) |

</div>

<br/>

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit | Interactive web interface |
| **Backend** | Python | Core application logic |
| **Machine Learning** | Scikit-learn | Model training & prediction |
| **NLP** | NLTK | Text preprocessing & tokenization |
| **Visualization** | Plotly | Interactive charts & dashboards |
| **Explainability** | SHAP | Feature attribution for predictions |
| **Deployment** | Streamlit Cloud | Hosting the live application |

<br/>

## 📂 Project Structure

```
ContentIQ/
├── Home.py
├── train.py
├── predict.py
├── config.py
├── requirements.txt
├── README.md
├── models/
├── pages/
├── utils/
├── assets/
├── data/
├── notebooks/
└── docs/
```

<br/>

## 📊 Model Performance

| Metric | Score |
|---|---|
| **Accuracy** | 0.00 |
| **Precision** | 0.00 |
| **Recall** | 0.00 |
| **F1 Score** | 0.00 |
| **ROC-AUC** | 0.00 |
| **Model Selected** | Placeholder |

<div align="center">
<img src="assets/confusion_matrix.png" alt="Confusion Matrix" width="450" />
</div>

<br/>

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/username/ContentIQ.git
cd ContentIQ

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Launch the Streamlit app
streamlit run Home.py
```

<br/>

## 🔮 Future Scope

- 🌐 Browser Extension
- ⚡ Real-Time Website Auditing
- 🌍 Multi-language Support
- 🤖 LLM-Powered Content Suggestions
- ☁️ Cloud Deployment
- 🔍 Advanced Explainable AI

<br/>

## 🤝 Contributing

Contributions are welcome. Fork the repo, create a feature branch, and open a pull request. Please keep changes focused and well-documented.

<br/>

## 📜 License

This project is licensed under the MIT License — you are free to use, modify, and distribute this software with attribution, provided the original copyright notice is retained.

<br/>

## 👩‍💻 Author

<div align="center">

**Your Name**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-083A4F?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-A58D66?style=for-the-badge&logo=github&logoColor=white)](https://github.com/username)
[![Email](https://img.shields.io/badge/Email-Contact-407E8C?style=for-the-badge&logo=gmail&logoColor=white)](mailto:your.email@example.com)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-C0D5D6?style=for-the-badge)](https://yourportfolio.com)

</div>
