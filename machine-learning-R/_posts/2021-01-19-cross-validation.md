---
layout: post-R
title:  "Cross-Validation(R)"
date:   2021-01-14 02:02:23 +0000
category: machine-learning-R
author: Arize Emmanuel
tag: very good tutorials
---

One of the finest techniques to check the generalization power of a machine learning model is to use ***Cross-validation techniques***. **Cross-validation** refers to a set of methods for measuring the performance of a given predictive model. It can be computationally expensive, because they involve fitting the same model multiple times using different subsets of the training data. Cross-validation techniques generally involves the following process:

1.  Divide the available data set into two sets namely training and testing (validation) data set.

2.  Train the model using the training set

3.  Test the effectiveness of the model on the the reserved sample (testing) of the data set and estimate the prediction error.

**cross-validation methods for assessing model performance includes,**

         Validation set approach (or data split)
         Leave One Out Cross Validation
         k-fold Cross Validation
         Repeated k-fold Cross Validation
         

<h3><b> Validation Set Approach</b></h3>
The validation set approach involves
         
     1.  randomly dividing the available data set into two parts namely,  training data set and validation data set.

     2.  Train the model on the training data set

     3.  The Trained model is then used to predict observations in the validation   set to test the generalization 
   
       ability of  the model when faced with new observations by calculating the prediction error.

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
cat("The advertising dataset has",nrow(marketing),'observations and',
       ncol(marketing),'features')

```

The advertising datasets has 200 observations and 4 features

<p>displaying the first four rows or observations of the dataset</p>
```r
head(marketing,4)
```

<img class="w3-center" src="{{'/assets/images/R/crossvalidation/rmarketing.jpg' |relative_url}}">

<p> The code below splits the data into training and testing set with 70% of the instances in the training set and 30% in the testing set</p>

```{r}
random_sample<-createDataPartition(marketing$sales,p = 0.7,list = FALSE)
training_set<-marketing[random_sample,]
testing_set<-marketing[-random_sample,]
```

<p> Let now fit a linear regression model to the dataset</p>
```{r}
model<-lm(sales~.,data=training_set)
```

<p>we now test the trained model on the testing set</p>

```{r}
prediction<-model %>% predict(testing_set)
```

<p>The code below calculates the mean absolut error (MAE), root mean square error (RMSE) and the R-Squre of the model based on the test set</p>

```{r}
data.frame( R2 = R2(prediction, testing_set$sales),
            RMSE = RMSE(prediction, testing_set$sales),
            MAE = MAE(prediction, testing_set$sales))
```
<img class="w3-center" src="{{'/assets/images/R/crossvalidation/vterror.jpg' |relative_url}}">

Using RMSE, the prediction error rate is calculated by dividing the RMSE by the average value of the outcome variable, which should be as small as possible

```{r}
RMSE(prediction, testing_set$sales)/mean(testing_set$sales)
```
0.09954645



<h4><b> NOTE</b></h4>
the validation set approach is only useful when a large data set is available. The model is trained on only a subset of the data set so it is possible the model will not be able to capture certain patterns or interesting information about data which are only present in the test data, leading to higher bias. The estimate of the test error rate can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set.



<h3><b> LEAVE ONE OUT CROSS VALIDATION- LOOCV</b></h3>

LOOCV is a special case of K-cross-validation where the number of folds equals the number of instances in the data set.It involves splitting the date set into two parts. However, instead of creating two subsets of comparable size, only a single data point is reserved as the test set.
The model is trained on the training set which consist of all the data points except the reserved point and compute the test error on the reserved data point. It repeats the process until each of the n data points has served as the test set and then avarage the n test errors.


<p>Let now implement LOOCV</p>

```{r}
loocv_data<-trainControl(method = 'LOOCV')
loocv_model<-train(sales~.,data=marketing,method='lm',trControl=loocv_data)
loocv_model
```
Although in LOOCV method, we make use all data points reducing potential bias, it is a poor estimate because it is highly variable, since it is based upon a single observation especially if some data points are outliers and has higher execution time when n is extremely large.



<h3><b> K-Fold Cross-Validation</b></h3> 

In practice if we have enough data, we set aside part of the data set known as the validation set and use it to measure the performance of our model prediction but since data are often scarce, this is usually not possible and the best practice in such situations is to use **K-fold cross-validation**.

<h4><b>K-fold cross-validation involves</b></h4> 

1.  Randomly splitting the data set into k-subsets (or k-fold) 
2. Train the model on K-1  subsets
3. Test the model on the reserved subset and record the prediction error
4. Repeat this process until each of the k subsets has served as the test set.
5. The average of the K validation scores is then obtained and used as the validation score for the model and is known as the cross-validation error .

```{r}
k_data<-trainControl(method = 'cv',number = 5)
cv_model<-train(sales~.,data=marketing,method='lm',trControl=k_data)
cv_model
```


<h3><b>  REPEATED K-FOLD CROSS-VALIDATION</b></h3>

The process of splitting the data into k-folds can be repeated a number of times, this is called repeated k-fold cross validation.

number -the number of folds 

repeats	For repeated k-fold cross-validation only: the number of complete sets of folds to compute

```{r}
train(sales~.,marketing , method="lm",trControl=trainControl(method ="repeatedcv",
            number = 5,repeats = 3),preProcess='scale')
```
<img class="w3-center" src="{{'/assets/images/R/crossvalidation/repeatedcv.jpg' |relative_url}}">

<h2> References:</h2>
- <a href="https://www.springer.com/series/417" target="_blank"> 
Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani. An Introduction to Statistical Learning  with Applications in R.</a><br>