#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize Otter
import otter
grader = otter.Notebook("proj2b.ipynb")


# # Project 2B: Spam/Ham Classification - Build Your Own Model
# 
# ## Feature Engineering, Classification, Cross Validation
# ## Due Date: Sunday 4/28, 11:59 PM PDT
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In this project, you will be building and improving on the concepts and functions that you implemented in Project 2A to create your own classifier to distinguish spam emails from ham (non-spam) emails. We will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this assignment, you should feel comfortable with the following:
# 
# - Using `sklearn` libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# ## Warning
# This is a **real world** dataset– the emails you are trying to classify are actual spam and legitimate emails. As a result, some of the spam emails may be in poor taste or be considered inappropriate. We think the benefit of working with realistic data outweighs these innapropriate emails, and wanted to give a warning at the beginning of the project so that you are made aware.

# In[2]:


# Run this cell to suppress all FutureWarnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ## Score Breakdown
# Question | Points
# --- | ---
# 1 | 6
# 2a | 4
# 2b | 2
# 3 | 3
# 4 | 15
# Total | 30

# ## Setup and Recap
# 
# Here we will provide a summary of Project 2A to remind you of how we cleaned the data, explored it, and implemented methods that are going to be useful for building your own model.

# In[3]:


import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading and Cleaning Data
# 
# Remember that in email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the unlabeled test set contains 1000 unlabeled examples.
# 
# Run the following cell to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to the autograder for evaluation.

# In[4]:


import zipfile
with zipfile.ZipFile('spam_ham_data.zip') as item:
    item.extractall()


# In[5]:


original_training_data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# Feel free to explore the dataset above along with any specific spam and ham emails that interest you. Keep in mind that our data may contain missing values, which are handled in the following cell.

# In[6]:


# Fill any missing or NAN values
print('Before imputation:')
print(original_training_data.isnull().sum())
original_training_data = original_training_data.fillna('')
print('------------')
print('After imputation:')
print(original_training_data.isnull().sum())


# ### Training/Validation Split
# 
# Recall that the training data we downloaded is all the data we have available for both training models and **validating** the models that we train. We therefore split the training data into separate training and validation datsets. You will need this **validation data** to assess the performance of your classifier once you are finished training. 
# 
# As in Project 2A, we set the seed (random_state) to 42. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[7]:


# This creates a 90/10 train-validation split on our labeled data
from sklearn.model_selection import train_test_split
train, val = train_test_split(original_training_data, test_size = 0.1, random_state = 42)

# We must do this in order to preserve the ordering of emails to labels for words_in_texts
train = train.reset_index(drop = True)


# ### Feature Engineering
# 
# In order to train a logistic regression model, we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$. To address this, in Project 2A, we implemented the function `words_in_texts`, which creates numeric features derived from the email text and uses those features for logistic regression. 
# 
# For this project, we have provided you with an implemented version of `words_in_texts`. Remember that the function outputs a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. 

# In[8]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    import numpy as np
    indicator_array = 1 * np.array([texts.str.contains(word) for word in words]).T
    return indicator_array


# Run the following cell to see how the function works on some dummy text.

# In[9]:


words_in_texts(['hello', 'bye', 'world'], pd.Series(['hello', 'hello worldhello']))


# ### EDA and Basic Classification
# 
# In Project 2A, we proceeded to visualize the frequency of different words for both spam and ham emails, and used `words_in_texts(words, train['email'])` to directly to train a classifier. We also provided a simple set of 5 words that might be useful as features to distinguish spam/ham emails. 
# 
# We then built a model using the using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier from `scikit-learn`.
# 
# Run the following cell to see the performance of a simple model using these words and the `train` dataframe.

# In[10]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = np.array(train['spam'])

X_train[:5], Y_train[:5]


# In[11]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver = 'lbfgs')
model.fit(X_train, Y_train)

training_accuracy = model.score(X_train, Y_train)
print("Training Accuracy: ", training_accuracy)


# ### Evaluating Classifiers

# In our models, we are evaluating accuracy on the training set, which may provide a misleading accuracy measure. In Project 2A, we calculated various metrics to lead us to consider more ways of evaluating a classifier, in addition to overall accuracy. Below is a reference to those concepts.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# To be clear, we label spam emails as 1 and ham emails as 0. These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The two graphics below may help you understand precision and recall visually:
# 
# ![precision_recall](precision_recall.png)
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# # Moving Forward - Building Your Own Model
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Gradescope.
# 
# **Gradescope limits you to four submissions per day**. You will be able to see your accuracy on the entire test set when submitting to Gradescope.
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!'s were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better (and/or more) words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# You may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No decision trees, random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 1, 2, and 3 should help guide you.
# 
# ---
# 
# **Note:** *You may want to use your **validation data** to evaluate your model and get a better sense of how it will perform on the test set.* Note, however, that you may overfit to your validation set if you try to optimize your validation accuracy too much. Alternatively, you can perform cross-validation on the entire training set.
# 
# ---

# <!-- BEGIN QUESTION -->
# 
# ### Question 1: Feature/Model Selection Process
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked or didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q1
# manual: True
# points: 6
# -->

# _The biggest step I took in improving my model was taking a deep dive into the ham and spam emails/subjects so as to idenify any major patterns that systematically distinguished the two email types. This mainly constisted of checking for words or regex patterns that more frequently existed in one type of email over the other. The words from the staff's bar chart were easy inclusions, but I also followed the above instructions for finding good features. I found it challenging to calculate the number of punctuation/words/characters in each email/subject, and even when I succeeded I often found that the feature did not systematically vary enough to impact model accuracy. This was surprising, as I expected the instructions wouldn't have been there if they weren't going to be impactful. During the rest of my editing I also noticed that in some cases, generalizing my features made them more impactful towards model accuracy. For example, instead of always creating different features for exact regex patterns, I sometimes combined features into a single feature that could take the value of several similar patterns. It seems as though in some cases, accuracy improved as features became more concise. However, in other cases, the notion that creating as many features as possible, even if they were similar, turned out to increase accuracy. This discrepancy made little sense to me. I eventually got my desired accuracy by using a scratch cell to figure out which features systematically took on different values for (sp)/ham emails. I calculated what percentage of each spam/ham group emails met a specific feature's condition. If the difference between the percentages was large, I put it in the features array to see how it impacted accuracy. My scratch cell consisted of word lists that led to training and validation data, which was used to fit and score different models. Simply changing the word lists allowed me to quickly see how they impacted both training and validation accuracy._

# <!-- END QUESTION -->
# 
# 
# 
# Optional: Build a Decision Tree model with reasonably good accuracy. What features does the decision tree use first?

# ### Question 2: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. 
# 
# Include:
# 
# 1. A plot showing something meaningful about the data that helped you during feature selection, model selection, or both.
# 2. Two or three sentences describing what you plotted and its implications with respect to your features.
# 
# Feel free to create as many plots as you want in your process of feature selection, but select only one for the response cell below.
# 
# **You should not just produce an identical visualization to Question 3 in Project 2A.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, **as long as it comes with thoughtful commentary.** Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# <!-- BEGIN QUESTION -->
# 
# #### Question 2a
# 
# Generate your visualization in the cell below.
# 
# <!--
# BEGIN QUESTION
# name: q2a
# manual: True
# format: image
# points: 4
# -->

# In[12]:


email_words_heat = ["<html>", "<body>", "<p>", "<head>", r"(^[a-z])\1{2,}", r"(m|M)oney", r"(p|P)lease", "(o|O)ffer",
               "(e|E)xpire", "<title>", "value=", "<option", "/", "color=", "height=", r".*t(d|r)", r"='.+'", r"_{3,}",
               "dear", "$", "face=", "vacation", r"(=.{2}){2,}", "href", "table"]
subj_words_heat = [r"(f|F)(ree|REE)", r"(R(e|E):)|(F(wd|WD):)", r"(.)\1{4,}", r"(y|Y)ou"]
X_heat_email = words_in_texts(email_words_heat, original_training_data["email"])
X_heat_subj = words_in_texts(subj_words_heat, original_training_data["subject"])
X_heat = np.hstack((X_heat_email, X_heat_subj))
heatmap_frame = pd.DataFrame(X_heat, columns=["e_HTML", "e_Body", "e_<p>", "e_<head>", "e_SymbolRepeat", "e_Money",
                "e_Please", "e_Offer", "e_Expire", "e_title", "e_value=", "e_<option", "e_/", "e_color=", "e_height=",
                "e_td/tr_", "e_=_", "e_underscore3x", "e_dear", "e_$", "e_face=", "e_vacation", "e_=pattern", "e_href",
                "e_table", "s_Free", "s_Re:/Fwd:", "s_Repeat4x", "s_You"])
heatmap_frame = heatmap_frame.join(original_training_data.loc[:, ["spam"]])
heatmap_frame = heatmap_frame.corr()
heatmap_frame = heatmap_frame[(heatmap_frame["spam"] >= 0.4) | (heatmap_frame["spam"] <= -0.4)]
heatmap_frame = heatmap_frame.loc[:, heatmap_frame.index.values]
sns.heatmap(heatmap_frame);


# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# #### Question 2b
# 
# Write your commentary in the cell below.
# 
# <!--
# BEGIN QUESTION
# name: q2b
# manual: True
# points: 2
# -->

# _To select features, I mostly used the process described in part 2a. However, to get a sense of which features most STRONGLY correlated with classification, I created this correlation heatmap using the features that had 0.4 or higher correlation with classification. Negatively-correlating variables were included. The map showed me that these features not only strongly correlated with classification, but they also correlated significantly with each other. This mostly applied to the HTML, color=, face=, and href features, as the presence of one of these features in an email often meant the rest were present as well. This makes sense in theory, as these features represent the presence of embedded code often associated with images or ads that usually only exist in spam emails. After seeing this trend, I tried testing my model using only these features, but the accuracy only reached 84% for both training and validation data. This told me that as important as these features were, they were not enough on their own to satisfactorily predict correct classifications._

# <!-- END QUESTION -->
# 
# <!-- BEGIN QUESTION -->
# 
# ### Question 3: ROC Curve
# 
# In most cases we won't be able to get 0 false positives and 0 false negatives, so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover that they have cancer until it's too late, whereas a patient can just receive another screening for a false positive.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The ROC curve shows this trade off for each possible cutoff probability. In the cell below, plot a ROC curve for your final classifier (the one you use to make predictions for Gradescope) on the training data. Refer to Lecture 20 to see how to plot an ROC curve.
# 
# **Hint**: You'll want to use the `.predict_proba` method for your classifier instead of `.predict` so you get probabilities instead of binary predictions.
# 
# <!--
# BEGIN QUESTION
# name: q3
# manual: True
# points: 3
# -->

# In[13]:


from sklearn.metrics import roc_curve

model_roc = LogisticRegression(fit_intercept=True, solver = "lbfgs")
email_words_roc = ["<html>", "<body>", "<p>", "<head>", r"(^[a-z])\1{2,}", r"(m|M)oney", r"(p|P)lease", "(o|O)ffer",
               "(e|E)xpire", "<title>", "value=", "<option", "/", "color=", "height=", r".*t(d|r)", r"='.+'", r"_{3,}",
               "dear", "$", "face=", "vacation", r"(=.{2}){2,}", "href", "table"]
subj_words_roc = [r"(f|F)(ree|REE)", r"(R(e|E):)|(F(wd|WD):)", r"(.)\1{4,}", r"(y|Y)ou"]
X_train_roc_email = words_in_texts(email_words_roc, train["email"])
X_train_roc_subj = words_in_texts(subj_words_roc, train["subject"])
X_train_roc, Y_train_roc = np.hstack((X_train_roc_email, X_train_roc_subj)), train["spam"].array
model_roc.fit(X_train_roc, Y_train_roc)

def predict_threshold(model, X, T): 
    prob_one = model.predict_proba(X)[:, 1]
    return (prob_one >= T).astype(int)

def tpr_threshold(X, Y, T):
    Y_hat = predict_threshold(model_roc, X, T)
    return np.sum((Y_hat == 1) & (Y == 1)) / np.sum(Y == 1)

def fpr_threshold(X, Y, T):
    Y_hat = predict_threshold(model_roc, X, T)
    return np.sum((Y_hat == 1) & (Y == 0)) / np.sum(Y == 0)

thresholds = np.linspace(0, 1, 100)
tprs = [tpr_threshold(X_train_roc, Y_train_roc, t) for t in thresholds]
fprs = [fpr_threshold(X_train_roc, Y_train_roc, t) for t in thresholds]

roc_frame = pd.DataFrame({"fprs": fprs, "tprs": tprs})
roc_plot = roc_frame.plot.line(x="fprs", y="tprs")
roc_plot.set_xlabel("False Positive Rate")
roc_plot.set_ylabel("True Positive Rate")
roc_plot.set_title("ROC Curve");


# <!-- END QUESTION -->
# 
# # Question 4: Test Predictions
# 
# The following code will write your predictions on the test dataset to a CSV file. **You will need to submit this file to the "k Test Predictions" assignment on Gradescope to get credit for this question.**
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. **Please make sure you've saved your predictions to `test_predictions` as this is how part of your score for this question will be determined.**
# 
# **Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions.** For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# **Note: You may submit up to 4 times a day. If you have submitted 4 times on a day, you will need to wait until the next day for more submissions.**
# 
# Note that this question is graded on an absolute scale based on the accuracy your model achieves on the overall test set, and as such, your score does not depend on your ranking on Gradescope.
# 
# *The provided tests check that your predictions are in the correct format, but you must additionally submit to Gradescope to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q4
# points: 3
# -->

# In[14]:


test = test.fillna('')
model_k = LogisticRegression(solver = "lbfgs")
email_words = ["<html>", "<body>", "<p>", "<head>", r"(^[a-z])\1{2,}", r"(m|M)oney", r"(p|P)lease", "(o|O)ffer",
               "(e|E)xpire", "<title>", "value=", "<option", "/", "color=", "height=", r".*t(d|r)", r"='.+'", r"_{3,}",
               "dear", "$", "face=", "vacation", r"(=.{2}){2,}", "href", "table"]
subj_words = [r"(f|F)(ree|REE)", r"(R(e|E):)|(F(wd|WD):)", r"(.)\1{4,}", r"(y|Y)ou"]
X_train_k_email, X_train_k_subj = words_in_texts(email_words, train["email"]), words_in_texts(subj_words, train["subject"])
X_test_k_email, X_test_k_subj = words_in_texts(email_words, test["email"]), words_in_texts(subj_words, test["subject"])
X_train_k, Y_train_k = np.hstack((X_train_k_email, X_train_k_subj)), train["spam"].array
X_test_k = np.hstack((X_test_k_email, X_test_k_subj))
model_k.fit(X_train_k, Y_train_k)
test_predictions = model_k.predict(X_test_k)


# In[15]:


grader.check("q4")


# The following cell generates a CSV file with your predictions. **You must submit this CSV file to the "Project 2B Test Predictions" assignment on Gradescope to get credit for this question.**
# 
# Note that the file will appear in your DataHub, you must navigate to the `hw11` directory in your DataHub to download the file.

# In[ ]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Gradescope for scoring.')


# ## Congratulations! You have completed Project 2B!

# ---
# 
# To double-check your work, the cell below will rerun all of the autograder tests.

# In[ ]:


grader.check_all()


# ## Submission
# 
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output. The cell below will generate a zip file for you to submit. **Please save before exporting!**

# In[ ]:


# Save your notebook first, then run this cell to export your submission.
grader.export()


#  
