# **PHASE 4 PROJECT: Natural Language Processing (NLP)**

## **Sentiment Analysis of Apple Tweets**

### **Understanding Public Perception: Sentiment Analysis of Apple Tweets**

In this project, I explored how people feel about Apple products by analyzing sentiment in tweets using Natural Language Processing (NLP). The dataset, sourced from CrowdFlower via data.world, contains **3,886 human-labeled tweets** categorized as positive, negative, or neutral. My goal was to build a machine learning model capable of accurately classifying tweet sentiment based on its content, allowing Apple to gauge customer reactions, monitor brand perception, and respond proactively to user feedback.

---

## 1. Business Understanding

### **Objective:**  
The goal of this project is to build an **NLP model** that can analyze **Twitter sentiment** regarding **Apple products**. The dataset contains **over 3,886 Tweets**, each labeled by human raters as **positive, negative, or neutral**. The model will classify new Tweets based on their sentiment, helping businesses understand **public perception and customer sentiment trends**.

### **Key Questions & Answers:**

#### **What is the goal of this project?**  
- To develop a **sentiment classification model** that can automatically **rate the sentiment of Tweets** related to Apple products.
- The model should be able to **identify positive, negative, or neutral sentiments** from textual data.
- The project aims to provide a **proof of concept**, starting with a simple **binary classifier (positive vs. negative)** and then expanding to a **multiclass classifier (positive, neutral, negative)**.

#### **What insights will be gained?**  
- **Brand Perception:** How do users feel about Apple products on social media?  
- **Customer Pain Points:** What negative aspects are frequently mentioned?  
- **Market Trends:** Are there specific events or product launches that trigger sentiment shifts?  
- **Influence of Tweet Content:** Which words or phrases contribute most to each sentiment category?  

#### **How will the results be used in decision-making?**  
- **Marketing Strategy:** Apple can adjust its marketing campaigns based on sentiment trends.  
- **Customer Support Prioritization:** If negative sentiment spikes, Apple can proactively address issues.  
- **Product Improvement:** Insights from common complaints can guide product development.  
- **Crisis Management:** Early detection of negative sentiment trends can help mitigate PR disasters.

---

## **2. Data Preparation**
Text preprocessing included:
- Removal of punctuation, special characters, and URLs.
- Tokenization and lowercasing of words.
- Stopword removal to reduce noise.
- Lemmatization for standardizing word forms.
- Feature extraction using TF-IDF Vectorization.

Python libraries used: `NLTK`, `scikit-learn`, and `pandas` for text processing and feature engineering.

---

## **3. Model Training**
- The dataset was split into **80% training and 20% testing**.
- Class imbalance was addressed using **SMOTE (Synthetic Minority Over-sampling Technique)**.
- The following models were trained and optimized:
  - **Naïve Bayes (MultinomialNB)** with hyperparameter tuning (`alpha` values: 0.1, 0.5, 1.0).
  - **Logistic Regression** with `MaxAbsScaler` normalization and hyperparameter tuning (`C` values: [0.0001, 0.001, 0.01, 0.1, 1]).
  - **Support Vector Machine (SVM)** with a linear kernel, `MaxAbsScaler`, and hyperparameter tuning (`C` values: [0.0001, 0.001, 0.01, 0.1, 1, 10]).

---

## **4. Model Evaluation**
### **Accuracy Scores**
- **Naïve Bayes:** 0.70
- **Logistic Regression:** 0.75
- **SVM:** 0.72

### **Classification Report (Logistic Regression)**
| Sentiment  | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| **Negative** | 0.71 | 0.73 | 0.72 |
| **Neutral** | 0.61 | 0.39 | 0.48 |
| **Positive** | 0.78 | 0.83 | 0.81 |
| **Macro Avg** | 0.70 | 0.65 | 0.67 |
| **Weighted Avg** | 0.74 | 0.75 | 0.74 |

### **Confusion Matrix**
- The **neutral class** had the lowest recall (0.39), indicating it was the hardest to classify.

### **AUC-ROC Score**
- **Logistic Regression AUC-ROC:** 0.84 (suggesting good separability despite lower accuracy).

---

## **5. Best Performing Model**
### **Logistic Regression**
#### **Reasons:**
1. **Highest Accuracy:** 0.75 (compared to 0.70 for Naïve Bayes and 0.72 for SVM).
2. **Strong Precision & Recall:** Particularly for the **positive** and **negative** classes.
3. **Best AUC-ROC Score:** 0.84, indicating better class separability.

However, the **neutral class** remains the hardest to classify (low recall of 0.39). 

---

## **6. Limitations & Recommendations**
### **Challenges**
- Handling ambiguous tweets, sarcasm, and irrelevant content remains difficult.

### **Future Improvements**
- Adjust class weights (`class_weight='balanced'`) in Logistic Regression.
- Experiment with **TF-IDF** instead of Count Vectorizer.
- Use an ensemble approach, such as **Logistic Regression + SVM**.
- Explore **deep learning** methods like LSTMs or transformers (BERT, RoBERTa).
- Deploy the model as an **API for real-time sentiment analysis**.

Through this project, I gained valuable insights into the power of NLP for sentiment classification, showcasing how businesses like Apple can leverage social media data to refine their strategies and better understand their customers. By refining model performance and expanding feature engineering, this approach can be scaled for broader applications in social media analysis.

## **7. Conclusion & Recommendations**

### **Conclusion**
This project successfully built and evaluated a sentiment analysis model that can classify tweets about Apple products as **positive, negative, or neutral** based on their content. After testing multiple models, **Logistic Regression** emerged as the best-performing approach, achieving **75% accuracy** and an **AUC-ROC score of 0.84**.  

Key findings:
- **Positive and negative tweets** were classified with good precision and recall.
- **Neutral tweets** were the hardest to classify accurately, with a recall of only 39%.
- **TF-IDF vectorization and Logistic Regression** provided a strong baseline model.

While the model performs well, challenges like **sarcasm, ambiguity, and short-text limitations** remain. Further improvements can help refine sentiment classification.

---

### **Recommendations**
To enhance model performance and achieve more accurate sentiment classification, the following steps are recommended:

1. **Improve Data Preprocessing**
   - Use **emoji and slang dictionaries** to capture informal language in tweets.
   - Implement **named entity recognition (NER)** to understand references to Apple products.
   - Expand the dataset with **more labeled examples**, especially for the **neutral** class, to improve classification balance.

2. **Feature Engineering Enhancements**
   - Explore **word embeddings (Word2Vec, GloVe, or BERT)** instead of TF-IDF for better contextual understanding.
   - Incorporate **bigram/trigram features** to capture phrase-based sentiment.

3. **Model Improvements**
   - Fine-tune **hyperparameters** and use **ensemble models** (e.g., stacking Logistic Regression and SVM).
   - Implement **deep learning models** such as **LSTMs, GRUs, or transformers (BERT, RoBERTa)**.

4. **Address Class Imbalance**
   - Apply **class weighting** to better handle underrepresented neutral tweets.
   - Use **oversampling techniques** like SMOTE or generate synthetic neutral examples.

5. **Deploying the Model**
   - Package the model as an **API** to analyze real-time tweets.
   - Integrate the model with **Apple's social media monitoring tools** to track sentiment trends.

By implementing these improvements, the sentiment analysis model can become more robust, accurately capturing how consumers feel about Apple products based on their tweets. This will enable Apple to **monitor brand perception, improve customer engagement, and enhance product strategies** based on real-time feedback.
