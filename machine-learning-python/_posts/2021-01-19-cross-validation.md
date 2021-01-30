---
layout: post-python
title:  "Cross-Validation (ML-python)"
date:   2021-01-14 02:02:23 +0000
category: machine-learning-python
author: Arize Emmanuel
---

One of the finest techniques to check the generalization power of a machine learning model is to use ***Cross-validation techniques***. **Cross-validation** refers to a set of methods for measuring the performance of a given predictive model and can be computationally expensive, because they involve fitting the same model multiple times using different subsets of the training data. Cross-validation techniques generally involves the following process:

1.  DividE the available data set into two sets namely training and testing (validation) data set.

2.  Train the model using the training set

3.  Test the effectiveness of the model on the the reserved sample (testing) of the data set and estimate the prediction error.

**cross-validation methods for assessing model performance includes,**

         Validation set approach (or data split)
         Leave One Out Cross Validation
         k-fold Cross Validation
         Repeated k-fold Cross Validation
         

# Validation Set Approach
The validation set approach involves
         
     1.  randomly dividing the available data set into two parts namely,  training data set and validation data set.

     2.  Model is trained on the training data set

     3.  The Trained model is then used to predict observations in the validation   set to test the generalization 
   
       ability of  the model when faced with new observations by calculating the prediction error using model 
       
       performance metrics
<p> importing the needed packages </p> 
```{python}
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,LeaveOneOut,KFold,RepeatedKFold
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer,mean_squared_error,r2_score
import sklearn
import pandas as pd

```
Loading the data 
```python
marketing=pd.read_csv('data/marking.csv',index_col=0)
print("The advertising datasets has "+
      str(marketing.shape[0])+ ' observations and '+str(marketing.shape[1])+' features')
```
The advertising datasets has 200 observations and 4 features

<p>displaying the first four rows or observations of the dataset</p>
```python
marketing.head(4)
```
<img class="w3-center" src="{{'/assets/python/images/marketing.jpg' |relative_url}}" style="width:300px;height:200px;">

### Feature Scaling or transformation

Numerical features are often measured on different scales and this variation in scale may pose problems to modelling the data correclty. With few exceptions, variations in scales of numerical features often lead to machine Learning algorithms not performing well. When features have different scales, features with higher magnitude are likely to have higher weights and this affects the performance of the model.

<em>Feature scaling is a technique applied as part of the data preparation process in machine learning to put the numerical features on a common scale without distorting the differences in the range of values.</em>
There are many feature scaling techniques such as

<p class="w3-panel w3-border w3-leftbar w3-center w3-rightbar w3-round-xlarge">
       <b> Min-max scaling (normalization)</b>
$$ x_{unit-interval}= \frac{x-min(x)}{max(x)-min(x)}$$
</p>

<p class="w3-panel w3-border w3-leftbar w3-center w3-rightbar w3-round-xlarge">
       <b>Box-Cox transformation.</b>
$$ x_{box-cox}= \frac{x^{\lambda}-1}{ \lambda }$$
</p>
      
<p class="w3-panel w3-border w3-leftbar w3-center w3-rightbar w3-round-xlarge">
       <b>Mean normalization.</b>
$$ x_{mean-normalization}= \frac{x-\bar{x}}{max(x)-min(x)}$$

</p>
      
 
 etc, but We will discuss only <b>Standardization(Z-score)</b> 
 
 ## Standardization(Z-score) 
When Z-score is applied to a feature this makes the feature have a zero mean by subtracting the mean of the feature from the data points and then it divides by the standard deviation so that the resulting distribution has unit variance.
 
 <p class="w3-panel w3-border w3-leftbar w3-center w3-rightbar w3-round-xlarge">
       <b>Standardization.</b>
$$ x_{z-score}= \frac{x-\bar{x}}{ \sigma}$$
</p>

since all features in our dataset are numrical we are going to scale them using Z-score

<p><b>Note: the target values is generally not scaled</b></p>

```python
scale=StandardScaler()
features=scale.fit_transform(marketing.iloc[:,:3])
features[0:3]
```
<img class="w3-center" src="{{'/assets/images/python/scale_market.jpg' |relative_url}}">