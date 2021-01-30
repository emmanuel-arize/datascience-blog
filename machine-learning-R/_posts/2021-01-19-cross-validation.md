---
layout: post-R
title:  "Cross-Validation(R)"
date:   2021-01-14 02:02:23 +0000
category: machine-learning-R
author: Arize Emmanuel
tag: very good tutorials
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
<p> loading the needed libraries </p> 
```{r}
library(tidyverse)
library(caret)
```
Loading the data 
```{r}
data("marketing", package = "datarium")

```
```{r}
cat("The advertising dataset has",nrow(marketing),'observations and',ncol(marketing),'features')

```

The advertising datasets has 200 observations and 4 features

<p>displaying the first four rows or observations of the dataset</p>
```r
head(marketing,4)
```

<img class="w3-center" src="{{'/assets/images/rmarketing.jpg' |relative_url}}">