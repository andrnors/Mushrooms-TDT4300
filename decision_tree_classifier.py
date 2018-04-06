
# coding: utf-8

# # Decision Tree Classifier
# 
# In this notebook, you will implement your own decision tree algorithm for the classification problem. You are supposed to learn:
# 
# * How to prepare the dataset for training and testing of the model (i.e. decision tree).
# * How to implement the decision tree learning algorithm.
# * How to classify unseen samples using your model (i.e. trained decision tree).
# * How to evaluate the performance of your model.
# 
# **Instructions:**
# 
# * Read carefuly through this notebook. Be sure you understand what is provided to you, and what is required from you.
# * Place your code/edit only in sections annotated with `### START CODE HERE ###` and `### END CODE HERE ###`.
# * Use comments whenever the code is not self-explanatory.
# * Submit an executable notebook (`*.ipynb`) with your solution to BlackBoard.
# 
# Enjoy :-)
# 
# ## Packages
# 
# Following packages is all you need. Do not import any additional packages!
# 
# * [Pandas](https://pandas.pydata.org/) is a library providing easy-to-use data structures and data analysis tools.
# * [Numpy](http://www.numpy.org/) library provides support for large multi-dimensional arrays and matrices, along with functions to operate on these.

# In[ ]:


import pandas as pd
import numpy as np


# ## Problem
# 
# You are given a dataset `mushrooms.csv` with characteristics/attributes of mushrooms, and your task is to implement, train and evaluate a decision tree classifier able to say whether a mushroom is poisonous or edible based on its attributes.
# 
# ## Dataset
# 
# The dataset of mushroom characteristics is freely available at [Kaggle Datasets](https://www.kaggle.com/uciml/mushroom-classification) where you can find further information about the dataset. It consists of 8124 mushrooms characterized by 23 attributes (including the class). Following is the overview of attributes and values:
# 
# * class: edible=e, poisonous=p
# * cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# * cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# * cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# * bruises: bruises=t,no=f
# * odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# * gill-attachment: attached=a,descending=d,free=f,notched=n
# * gill-spacing: close=c,crowded=w,distant=d
# * gill-size: broad=b,narrow=n
# * gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# * stalk-shape: enlarging=e,tapering=t
# * stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# * stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# * stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# * stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# * stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# * veil-type: partial=p,universal=u
# * veil-color: brown=n,orange=o,white=w,yellow=y
# * ring-number: none=n,one=o,two=t
# * ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# * spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# * population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# * habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d
# 
# Let's load the dataset into so called Pandas dataframe.

# In[ ]:


mushrooms_df = pd.read_csv('mushrooms.csv')


# Now we can take a closer look at the data.

# In[ ]:


mushrooms_df


# You can also print an overview of all attributes with the counts of unique values.

# In[ ]:


mushrooms_df.describe().T


# The dataset is pretty much balanced. That's a good news for the evaluation.

# ## Dataset Preprocessing
# 
# As our dataset consist of nominal/categorical values only, we will encode the strings into integers which again should simplify our implementation.

# In[ ]:


def encode_labels(df):
    import sklearn.preprocessing
    encoder = {}
    for col in df.columns:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoder[col] = le
    return df, encoder    

mushrooms_encoded_df, encoder = encode_labels(mushrooms_df)


# In[ ]:


mushrooms_encoded_df


# ## Dataset Splitting
# 
# Before we start with the implementation of our decision tree algorithm we need to prepare our dataset for the training and testing.
# 
# First, we divide the dataset into attributes (often called features) and classes (often called targets). Keeping attributes and classes separately is a common practice in many implementations. This should simplify the implementation and make the code understandable.

# In[ ]:


X_df = mushrooms_encoded_df.drop('class', axis=1)  # attributes
y_df = mushrooms_encoded_df['class']  # classes
X_array = X_df.as_matrix()
y_array = y_df.as_matrix()


# And this is how it looks like.

# In[ ]:


print('X =', X_array)
print('y =', y_array)


# Next, we need to split the attributes and classes into training sets and test sets.
# 
# **Exercise:**
# 
# Implement the holdout splitting method with shuffling.

# In[ ]:


def train_test_split(X, y, test_size=0.2):
    """
    Shuffles the dataset and splits it into training and test sets.
    
    :param X
        attributes
    :param y
        classes
    :param test_size
        float between 0.0 and 1.0 representing the proportion of the dataset to include in the test split
    :return
        train-test splits (X-train, X-test, y-train, y-test)
    """
    ### START CODE HERE ###

    ### END CODE HERE ###
    return X_train, X_test, y_train, y_test


# Let's split the dataset into training and validation/test set with 67:33 split.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)


# In[ ]:


print('X_train =', X_train)
print('y_train =', y_train)
print('X_test =', X_test)
print('y_test =', y_test)


# A quick sanity check...

# In[ ]:


assert len(X_train) == len(y_train)
assert len(y_train) == 5443
assert len(X_test) == len(y_test)
assert len(y_test) == 2681


# ## Training
# 
# **Exercise:**
# 
# Implement an algorithm for fitting (also called training or inducing) a decision tree.
# 
# * You have a free hand regarding the generation of candidate splits (also called attribute test conditions).
# * Measure the degree of impurity (Gini) to select the best split.

# In[ ]:


# Use this section to place any "helper" code for the `fit()` function.

### START CODE HERE ###

### END CODE HERE ###


# In[ ]:


def fit(X, y):
    """
    Function implementing decision tree induction.
    
    :param X
        attributes
    :param y
        classes
    :return
        trained decision tree (model)
    """
    ### START CODE HERE ### 

    ### END CODE HERE ### 
    return classifier


# In[ ]:


model = fit(X_train, y_train)


# ## Prediction/Deduction
# 
# At this moment we should have trained a decision tree (our model). Now we need an algorithm for assigning a class given the attributes and our model.
# 
# **Exercise:**
# 
# Implement an algorithm deducing class given the attributes and the model.
# 
# * `X` is a matrix of attributes of one or more instances for classification.

# In[ ]:


# Use this section to place any "helper" code for the `predict()` function.

### START CODE HERE ###

### END CODE HERE ###


# In[ ]:


def predict(X, model):
    """
    Function for generating predictions (classifying) given attributes and model.
    
    :param X
        attributes
    :param model
        model
    :return
        predicted classes (y_hat)
    """
    ### START CODE HERE ###

    ### END CODE HERE ###
    return y_hat


# Let's classify the instances of our test set.

# In[ ]:


y_hat = predict(X_test, model)


# First ten predictions of the test set.

# In[ ]:


y_hat[:10]


# ## Evaluation
# 
# Now we would like to assess how well our decision tree classifier performs.
# 
# **Exercise:**
# 
# Implement a function for calculating the accuracy of your predictions given the ground truth and predictions.

# In[ ]:


def evaluate(y_true, y_pred):
    """
    Function calculating the accuracy of the model given the ground truth and predictions.
    
    :param y_true
        true classes
    :param y_pred
        predicted classes
    :return
        accuracy
    """
    ### START CODE HERE ###

    ### END CODE HERE ### 
    return accuracy


# In[ ]:


accuracy = evaluate(y_test, y_hat)
print('accuracy =', accuracy)


# How many items where misclassified?

# In[ ]:


print('misclassified =', sum(abs(y_hat - y_test)))


# How balanced is our test set?

# In[ ]:


np.bincount(y_test)


# If it's balanced, we don't have to be worried about objectivity of the accuracy metric.

# ---
# 
# Congratulations! At this point, hopefully, you have successufuly implemented a decision tree algorithm able to classify unseen samples with high accuracy.
# 
# ✌️
