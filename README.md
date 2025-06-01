# Sentiment-Analysis-on-dataset-of-customer-reviews-using-TF-IDF-vectorization-

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: AISHWARYA SANGRAM MULE

*INTERN ID*: 

*DOMAIN*: MACHINE LEARNING

*DURATION*: 12 WEEKS

*MENTOR*: NEELA SANTOSH


##Sentiment Analysis using TF-IDF and Logistic Regression

Project Overview

This project demonstrates a basic implementation of sentiment analysis on a collection of user reviews using TF-IDF vectorization and Logistic Regression, two widely used techniques in Natural Language Processing (NLP) and Machine Learning. The goal is to classify text reviews as either positive (1) or negative (0) based on their content.

The dataset consists of 40 manually curated synthetic review sentences with an equal distribution of sentiments (20 positive and 20 negative). This dataset simulates realistic feedback that a product or service might receive, covering a variety of emotions and expressions commonly found in customer reviews.

Problem Statement

In today's digital era, customers share their experiences in the form of online reviews. These reviews can significantly influence the reputation of products and companies. Manually analyzing thousands of such reviews is time-consuming and inefficient. Hence, automated sentiment analysis has become crucial in fields like marketing, customer service, and product development.

This project aims to demonstrate how even a relatively simple machine learning model like logistic regression can provide valuable insights when paired with the right feature extraction technique such as TF-IDF (Term Frequency–Inverse Document Frequency).

Methodology

The implementation follows these key steps:

1. Dataset Creation:
A custom dataset containing 40 labeled reviews is created. Each review is labeled as either 1 (positive sentiment) or 0 (negative sentiment). The dataset includes varied phrasing to simulate real-world text diversity.


2. Data Splitting:
The data is split into training and testing sets using an 70-30 ratio via train_test_split() from Scikit-learn.


3. Feature Extraction:
Text reviews are converted into numerical feature vectors using TF-IDF Vectorizer, which helps in capturing the importance of words relative to the dataset.


4. Model Training:
A Logistic Regression model is trained using the transformed TF-IDF features of the training data.


5. Prediction & Evaluation:
The model makes predictions on the test data. Performance is evaluated using:

Accuracy Score

Classification Report (Precision, Recall, F1-Score)


Tools and Libraries

Python

Pandas

Scikit-learn

TfidfVectorizer

LogisticRegression

train_test_split

accuracy_score, classification_report, confusion_matrix

Results and Observations

Despite the small dataset size, the logistic regression model performs well due to the clear structure and balance in the data. TF-IDF helps in filtering out common, less informative words and highlights more meaningful words that help distinguish between positive and negative sentiments.

This project proves that with proper preprocessing and modeling, effective sentiment analysis can be achieved even with basic tools and minimal data.


Future Scope

Integrate a larger real-world dataset such as IMDb or Amazon reviews.

Apply advanced models like Naive Bayes, SVM, or neural networks (LSTM, BERT).

#OUTPUT

Deploy as a web application with live input support.
