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

## Usage

Simply run the main analysis notebook:

```bash
jupyter notebook code.ipynb
```

The notebook includes:

- Data scraping (using scrape.py)
- PDF processing and text extraction
- Data cleaning and preprocessing
- Feature engineering with TF-IDF
- Model training and evaluation
- SHAP analysis and visualization
- Word cloud generation

---

## 📊 Dataset

- **Total Cases:** 61,127 non-empty complaint records
- **Time Period:** 2013-2023
- **Features:** Complaint text, company, date, insurance type, decision outcome
- **Class Distribution:** Balanced representation of upheld vs. not upheld decisions
- **Source:** UK Financial Ombudsman Service public decisions


## 🔍 Model Interpretation

The project includes comprehensive SHAP analysis to explain model predictions:

- Top predictive features identified and visualized
- Feature importance across different complaint types
- Violin plots showing SHAP value distributions
- Noun-focused analysis for business interpretability



## 🎯 Results and Impact

### Key Findings

- **Most Predictive Features:** Identified top 3 noun features driving predictions
- **Business Insights:** Revealed patterns in complaint language that correlate with outcomes
- **Model Robustness:** Consistent performance across cross-validation folds

### Visualizations Generated

- Comparative word clouds for upheld vs. not upheld complaints
- SHAP summary plots for feature importance
- Model performance comparison charts
- Feature distribution analysis



## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.



## 🙏 Acknowledgments

UK Financial Ombudsman Service for making complaint data publicly available  
Open source community for the excellent Python data science libraries

> **Note:** This project is for educational and research purposes. Always ensure compliance with data usage policies and regulations when working with real-world complaint data.

