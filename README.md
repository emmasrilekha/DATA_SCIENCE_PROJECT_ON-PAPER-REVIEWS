# DATA_SCIENCE_PROJECT_ON-PAPER-REVIEWS
# Sentiment Analysis on Paper Reviews Using Machine Learning and Deep Learning

## Overview
This project applies classical and deep learning models to perform sentiment classification on academic peer reviews. The dataset includes multilingual reviews (Spanish and English) with numerical sentiment scores mapped to Positive, Neutral, and Negative classes.

## Objectives
- Accurately classify peer reviews into sentiment categories.
- Compare performance across multiple models (Naïve Bayes, Logistic Regression, SVM, LSTM, BiLSTM).
- Handle challenges such as class imbalance and multilingual text.
- Incorporate pretrained fastText embeddings for semantic enhancement.

## Dataset
**Source:** [UCI Machine Learning Repository – Paper Reviews Dataset](https://archive.ics.uci.edu/dataset/410/paper+reviews)  
**Features:**
- `text`: Full review content  
- `evaluation`: Sentiment score (-2 to 2)  
- `language`: Language of the review (es, en)  
- `preliminary_decision`, `confidence`, `orientation`, `remarks`, etc.

## Models Used
- **Naïve Bayes** (Multinomial)
- **Logistic Regression** (with GridSearchCV tuning)
- **Linear SVM** (with hyperparameter tuning)
- **LSTM** (with sequence padding and tokenization)
- **BiLSTM** (enhanced with pretrained fastText embeddings)

## Methodology
- **Preprocessing:** Language translation, text normalization, tokenization
- **Feature Engineering:** TF-IDF for classical models; embeddings for deep models
- **Balancing:** Random upsampling on training data
- **Evaluation Metrics:** Accuracy, Precision, Recall, Macro F1-score
- **Tools:** Python, scikit-learn, TensorFlow, fastText, matplotlib, seaborn

## Results
| Model               | Accuracy | Macro F1 |
|---------------------|----------|----------|
| Naïve Bayes         | 0.85     | 0.86     |
| Logistic Regression | 0.95     | 0.95     |
| Linear SVM          | 0.95     | 0.95     |
| LSTM                | 0.94     | 0.94     |
| BiLSTM + fastText   | 0.94     | 0.94     |

## Key Findings
- BiLSTM with fastText achieved the best contextual performance.
- Classical models were faster and performed well with TF-IDF.
- Neutral class benefited most from upsampling.

## Future Work
- Integrate transformer-based models (e.g., DistilBERT, XLM-R)
- Apply domain-specific embeddings
- Add explainability tools (e.g., LIME, SHAP)

## License
This project is licensed under the MIT License.


