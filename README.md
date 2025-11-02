# Insurance Complaint Analysis System

## 📊 Project Overview

A comprehensive machine learning pipeline for analyzing insurance complaint decisions from the UK Financial Ombudsman Service. This system processes 61,127 complaint cases to predict whether complaints will be upheld or not upheld, providing valuable insights for insurance companies and regulatory bodies.

## 🚀 Key Features

- **End-to-End ML Pipeline**: From raw PDF documents to actionable insights
- **Advanced Text Processing**: NLP-powered feature extraction from complaint text
- **Multiple Model Comparison**: Comprehensive evaluation of various classifiers
- **SHAP Interpretability**: SHAP (SHapley Additive exPlanations) analysis to interpret model predictions
- **Interactive Visualizations**: Word clouds and feature importance charts

## 🏆 Model Performance

### Optimal Model: XGBClassifier

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.82 |
| **Macro Avg Precision** | 0.80 |
| **Macro Avg Recall** | 0.79 |
| **Macro Avg F1-Score** | 0.80 |
| **Precision (Class 0 - Not Upheld)** | 0.88 |
| **Precision (Class 1 - Upheld)** | 0.70 |
| **Mean CV Accuracy** | 0.8206 |
| **Std in CV** | 0.0023 |

### Model Comparison Summary
**XGBClassifier** leads in overall accuracy (0.82), macro average Precision/Recall/F1 (0.80/0.79/0.80) and cross-validation mean (0.8206), while maintaining reasonable minority class (Class 1) precision (0.70). RandomForest performs best on majority class (Class 0) with 0.91 precision but struggles with minority class (0.56). Logistic Regression shows balanced but moderate performance with minimal variance.

## 🛠 Technical Stack

### Data Processing
- **Web Scraping**: `requests`, `BeautifulSoup`
- **PDF Processing**: `PyMuPDF`, `pdfplumber`
- **Data Manipulation**: `pandas`, `numpy`

### Machine Learning & NLP
- **Text Vectorization**: `TF-IDF Vectorizer`
- **Classification**: `XGBoost`, `RandomForest`, `Logistic Regression`
- **NLP**: `spaCy`, `nltk`
- **Model Interpretation**: `SHAP`

### Visualization
- **Word Clouds**: `wordcloud`
- **Plots & Charts**: `matplotlib`, `seaborn`

## 📁 Project Structure


```text
insurance-complaint-analysis/
├── scrape.py                    # Data collection & web scraping script
├── analysis.ipynb               # Complete ML pipeline notebook
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── data/                        # Generated during execution
│   ├── raw/                     # Downloaded PDF documents
│   ├── processed/               # Cleaned and structured data
│   └── metadata.csv             # Complaint metadata (auto-generated)
│
├── models/                      # Generated during execution  
│   └── xgb_optimal.pkl          # Trained XGBoost model
│
└── results/                     # Generated during execution
    ├── wordclouds/              # NLP visualization outputs
    ├── shap_analysis/           # Model interpretation results
    └── performance_metrics/     # Evaluation metrics and charts

```

## 🎯 Business Value

This system helps:

- **Insurance Companies**: Identify patterns in successful vs. unsuccessful complaints
- **Regulatory Bodies**: Monitor complaint trends and handling effectiveness
- **Consumers**: Understand factors that influence complaint outcomes
- **Risk Management**: Proactively address common complaint triggers

## 📈 Key Insights

### Text Analysis Findings
- Distinct vocabulary patterns between upheld and not upheld complaints
- Specific noun phrases strongly predictive of complaint outcomes
- Clear linguistic markers that differentiate successful complaints

### Model Interpretation
- Top features identified through SHAP analysis provide actionable insights
- Noun-based features show highest predictive power
- Model maintains good performance on minority class (upheld complaints)

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/s2520758/insurance-complaint-analysis.git
cd insurance-complaint-analysis

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
