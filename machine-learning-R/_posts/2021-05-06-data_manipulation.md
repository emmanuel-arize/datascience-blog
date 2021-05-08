---
layout: post-R
title: Data Manipulation in R
category: machine-learning-R
---


This tutorial introduces how to easily manipulate data in R using the tidyverse package. Data manipulation usually involves

* computing summary statistics.

* rows filtering (filter()) and  ordering (arrange()).

* renaming (rename()) and selecting certain columns (select()

<b>and adding columns</b>

* mutate(): compute and add new variables into a data table. It preserves existing variables and adds new columns at the end of your dataset and 

* transmute(): compute new columns and only keep the new columns,




setwd("wdirectory") Changes the current working directory to wdirectory.


```
setwd('C:/Users/USER/Desktop/JUPYTER_NOTEBOOK/A_MYTUTORIALS/MYR')
```


Load the needed packages


```
library(devtools)
library(tidyverse)
library(nycflights13)
library(readxl)
```

    Loading required package: usethis
    
    Warning message in (function (kind = NULL, normal.kind = NULL, sample.kind = NULL) :
    "non-uniform 'Rounding' sampler used"
    -- [1mAttaching packages[22m ------------------------------------------------------------------------------- tidyverse 1.3.1 --
    
    [32mv[39m [34mggplot2[39m 3.3.3     [32mv[39m [34mpurrr  [39m 0.3.4
    [32mv[39m [34mtibble [39m 3.1.1     [32mv[39m [34mdplyr  [39m 1.0.5
    [32mv[39m [34mtidyr  [39m 1.1.3     [32mv[39m [34mstringr[39m 1.4.0
    [32mv[39m [34mreadr  [39m 1.4.0     [32mv[39m [34mforcats[39m 0.5.1
    
    Warning message in (function (kind = NULL, normal.kind = NULL, sample.kind = NULL) :
    "non-uniform 'Rounding' sampler used"
    -- [1mConflicts[22m ---------------------------------------------------------------------------------- tidyverse_conflicts() --
    [31mx[39m [34mdplyr[39m::[32mfilter()[39m masks [34mstats[39m::filter()
    [31mx[39m [34mdplyr[39m::[32mlag()[39m    masks [34mstats[39m::lag()
    
    

# Datatset
 We will use titanic dataset. This dataset has 1309 observations with 14 variables. To explore the basic data manipulation verbs of dplyr,
 we start by converting the data into a tibble data frame for easier data manipulation


```
my_data<-as_tibble(read_xls('data/titanic.xls'))
names(my_data)
```

    Warning message in read_fun(path = enc2native(normalizePath(path)), sheet_i = sheet, :
    "Coercing text to numeric in M1306 / R1306C13: '328'"



# Variable Selection 

we can select or subset variables by names or position.

Under variable selection we will learn how to use

* select(): allow us to extract variables or variables as a data table and can  also be used to remove variables from the data frame.

* select_if(): Select variabless based on a particular condition. 


* Variabl Selection by position

select from the my_data variable positioned from 1 to 4 inclusive



```
head(my_data %>% select(1:4))
```


<table class="dataframe">
<caption>A tibble: 6 × 4</caption>
<thead>
	<tr><th scope=col>pclass</th><th scope=col>survived</th><th scope=col>name</th><th scope=col>sex</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>1</td><td>Allen, Miss. Elisabeth Walton                  </td><td>female</td></tr>
	<tr><td>1</td><td>1</td><td>Allison, Master. Hudson Trevor                 </td><td>male  </td></tr>
	<tr><td>1</td><td>0</td><td>Allison, Miss. Helen Loraine                   </td><td>female</td></tr>
	<tr><td>1</td><td>0</td><td>Allison, Mr. Hudson Joshua Creighton           </td><td>male  </td></tr>
	<tr><td>1</td><td>0</td><td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td><td>female</td></tr>
	<tr><td>1</td><td>1</td><td>Anderson, Mr. Harry                            </td><td>male  </td></tr>
</tbody>
</table>




select variables positioned at 1, 4,6,7



```

head(my_data %>% select(1,4,6,7))

```


<table class="dataframe">
<caption>A tibble: 6 × 4</caption>
<thead>
	<tr><th scope=col>pclass</th><th scope=col>sex</th><th scope=col>sibsp</th><th scope=col>parch</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>female</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>male  </td><td>1</td><td>2</td></tr>
	<tr><td>1</td><td>female</td><td>1</td><td>2</td></tr>
	<tr><td>1</td><td>male  </td><td>1</td><td>2</td></tr>
	<tr><td>1</td><td>female</td><td>1</td><td>2</td></tr>
	<tr><td>1</td><td>male  </td><td>0</td><td>0</td></tr>
</tbody>
</table>



# Select varaibles by name


```
head(select(my_data,pclass,name:age),3)
```


<table class="dataframe">
<caption>A tibble: 3 × 4</caption>
<thead>
	<tr><th scope=col>pclass</th><th scope=col>name</th><th scope=col>sex</th><th scope=col>age</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>Allen, Miss. Elisabeth Walton </td><td>female</td><td>29.0000</td></tr>
	<tr><td>1</td><td>Allison, Master. Hudson Trevor</td><td>male  </td><td> 0.9167</td></tr>
	<tr><td>1</td><td>Allison, Miss. Helen Loraine  </td><td>female</td><td> 2.0000</td></tr>
</tbody>
</table>



Select all variables except variables from survived to cabin 


```
head(select(my_data,-(survived:cabin)),3)
```


<table class="dataframe">
<caption>A tibble: 3 × 5</caption>
<thead>
	<tr><th scope=col>pclass</th><th scope=col>embarked</th><th scope=col>boat</th><th scope=col>body</th><th scope=col>home.dest</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>S</td><td>2 </td><td>NA</td><td>St Louis, MO                   </td></tr>
	<tr><td>1</td><td>S</td><td>11</td><td>NA</td><td>Montreal, PQ / Chesterville, ON</td></tr>
	<tr><td>1</td><td>S</td><td>NA</td><td>NA</td><td>Montreal, PQ / Chesterville, ON</td></tr>
</tbody>
</table>




select variables whose name starts with bo


```
head(my_data %>% select(starts_with('bo')),3)
```


<table class="dataframe">
<caption>A tibble: 3 × 2</caption>
<thead>
	<tr><th scope=col>boat</th><th scope=col>body</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>2 </td><td>NA</td></tr>
	<tr><td>11</td><td>NA</td></tr>
	<tr><td>NA</td><td>NA</td></tr>
</tbody>
</table>



select variables whose name ends with t


```
head(my_data %>% select(ends_with('t')),3)
```


<table class="dataframe">
<caption>A tibble: 3 × 3</caption>
<thead>
	<tr><th scope=col>ticket</th><th scope=col>boat</th><th scope=col>home.dest</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>24160 </td><td>2 </td><td>St Louis, MO                   </td></tr>
	<tr><td>113781</td><td>11</td><td>Montreal, PQ / Chesterville, ON</td></tr>
	<tr><td>113781</td><td>NA</td><td>Montreal, PQ / Chesterville, ON</td></tr>
</tbody>
</table>




Select variables whose names contains "me"



```
head(my_data %>% select(contains('me')),4)
```


<table class="dataframe">
<caption>A tibble: 4 × 2</caption>
<thead>
	<tr><th scope=col>name</th><th scope=col>home.dest</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Allen, Miss. Elisabeth Walton       </td><td>St Louis, MO                   </td></tr>
	<tr><td>Allison, Master. Hudson Trevor      </td><td>Montreal, PQ / Chesterville, ON</td></tr>
	<tr><td>Allison, Miss. Helen Loraine        </td><td>Montreal, PQ / Chesterville, ON</td></tr>
	<tr><td>Allison, Mr. Hudson Joshua Creighton</td><td>Montreal, PQ / Chesterville, ON</td></tr>
</tbody>
</table>



# Variable selection based on a condtion

select only character variables


```
head(my_data %>% select_if(is.character),4)
```


<table class="dataframe">
<caption>A tibble: 4 × 7</caption>
<thead>
	<tr><th scope=col>name</th><th scope=col>sex</th><th scope=col>ticket</th><th scope=col>cabin</th><th scope=col>embarked</th><th scope=col>boat</th><th scope=col>home.dest</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Allen, Miss. Elisabeth Walton       </td><td>female</td><td>24160 </td><td>B5     </td><td>S</td><td>2 </td><td>St Louis, MO                   </td></tr>
	<tr><td>Allison, Master. Hudson Trevor      </td><td>male  </td><td>113781</td><td>C22 C26</td><td>S</td><td>11</td><td>Montreal, PQ / Chesterville, ON</td></tr>
	<tr><td>Allison, Miss. Helen Loraine        </td><td>female</td><td>113781</td><td>C22 C26</td><td>S</td><td>NA</td><td>Montreal, PQ / Chesterville, ON</td></tr>
	<tr><td>Allison, Mr. Hudson Joshua Creighton</td><td>male  </td><td>113781</td><td>C22 C26</td><td>S</td><td>NA</td><td>Montreal, PQ / Chesterville, ON</td></tr>
</tbody>
</table>




selecting only numerical variables


```
head(select_if(my_data,is.numeric),4)
```


<table class="dataframe">
<caption>A tibble: 4 × 7</caption>
<thead>
	<tr><th scope=col>pclass</th><th scope=col>survived</th><th scope=col>age</th><th scope=col>sibsp</th><th scope=col>parch</th><th scope=col>fare</th><th scope=col>body</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>1</td><td>29.0000</td><td>0</td><td>0</td><td>211.3375</td><td> NA</td></tr>
	<tr><td>1</td><td>1</td><td> 0.9167</td><td>1</td><td>2</td><td>151.5500</td><td> NA</td></tr>
	<tr><td>1</td><td>0</td><td> 2.0000</td><td>1</td><td>2</td><td>151.5500</td><td> NA</td></tr>
	<tr><td>1</td><td>0</td><td>30.0000</td><td>1</td><td>2</td><td>151.5500</td><td>135</td></tr>
</tbody>
</table>



# Removing columns


For simplicity we will work with only few variables 


```
sub_data<-my_data %>% select(sex,pclass,age,'survived',cabin,'name','sex','age','sibsp')
```

remove variables named pclass,age and sex



```
head(sub_data%>% select(-sex,-pclass,-age),3)
```


<table class="dataframe">
<caption>A tibble: 3 × 4</caption>
<thead>
	<tr><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>B5     </td><td>Allen, Miss. Elisabeth Walton </td><td>0</td></tr>
	<tr><td>1</td><td>C22 C26</td><td>Allison, Master. Hudson Trevor</td><td>1</td></tr>
	<tr><td>0</td><td>C22 C26</td><td>Allison, Miss. Helen Loraine  </td><td>1</td></tr>
</tbody>
</table>



# Rows filtering (filter()) 

This section describes how to subset or extract samples or rows from the dataset based on certain criteria

extract male (sex=='male') passengers who survived (survived==1)  and has sibsp==1 (Number of Siblings/Spouses Aboard)



```
head(sub_data %>% filter(sex=='male' & sibsp==1 & survived==1),2)
```


<table class="dataframe">
<caption>A tibble: 2 × 7</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>pclass</th><th scope=col>age</th><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>male</td><td>1</td><td> 0.9167</td><td>1</td><td>C22 C26</td><td>Allison, Master. Hudson Trevor</td><td>1</td></tr>
	<tr><td>male</td><td>1</td><td>37.0000</td><td>1</td><td>D35    </td><td>Beckwith, Mr. Richard Leonard </td><td>1</td></tr>
</tbody>
</table>



OR


```
head(sub_data %>% filter(sex=='male', sibsp==1,survived==1),2)
```


<table class="dataframe">
<caption>A tibble: 2 × 7</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>pclass</th><th scope=col>age</th><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>male</td><td>1</td><td> 0.9167</td><td>1</td><td>C22 C26</td><td>Allison, Master. Hudson Trevor</td><td>1</td></tr>
	<tr><td>male</td><td>1</td><td>37.0000</td><td>1</td><td>D35    </td><td>Beckwith, Mr. Richard Leonard </td><td>1</td></tr>
</tbody>
</table>



extract rows where  passengers are male(sex=='male') or survived (survived==1)  or has sibsp==1 or 2 (Number of Siblings/Spouses Aboard)



```
head(sub_data %>% filter(sex=='male' | sibsp==1 |sibsp==2| survived==1),3)
```


<table class="dataframe">
<caption>A tibble: 3 × 7</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>pclass</th><th scope=col>age</th><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>female</td><td>1</td><td>29.0000</td><td>1</td><td>B5     </td><td>Allen, Miss. Elisabeth Walton </td><td>0</td></tr>
	<tr><td>male  </td><td>1</td><td> 0.9167</td><td>1</td><td>C22 C26</td><td>Allison, Master. Hudson Trevor</td><td>1</td></tr>
	<tr><td>female</td><td>1</td><td> 2.0000</td><td>0</td><td>C22 C26</td><td>Allison, Miss. Helen Loraine  </td><td>1</td></tr>
</tbody>
</table>




select variables sibsp,sex,age and from these variables extract rows where age<10



```
head(sub_data %>% select(sex,sibsp,age) %>%filter(age<10),3)

```


<table class="dataframe">
<caption>A tibble: 3 × 3</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>sibsp</th><th scope=col>age</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>male  </td><td>1</td><td>0.9167</td></tr>
	<tr><td>female</td><td>1</td><td>2.0000</td></tr>
	<tr><td>male  </td><td>0</td><td>4.0000</td></tr>
</tbody>
</table>



# Selecting random rows or samples from a dataset

selecting 10 random samples without replacement from the data


```
head(sub_data %>% sample_n(10,replace = FALSE),2)
```


<table class="dataframe">
<caption>A tibble: 2 × 7</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>pclass</th><th scope=col>age</th><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>male  </td><td>3</td><td>20</td><td>0</td><td>NA  </td><td>Vendel, Mr. Olof Edvin                      </td><td>0</td></tr>
	<tr><td>female</td><td>1</td><td>35</td><td>1</td><td>C123</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td>1</td></tr>
</tbody>
</table>



Select 1% random samples without replacement from the data


```
sub_data %>% sample_frac(0.01,replace = FALSE)
```


<table class="dataframe">
<caption>A tibble: 13 × 7</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>pclass</th><th scope=col>age</th><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>male  </td><td>1</td><td>34</td><td>1</td><td>NA </td><td>Seward, Mr. Frederic Kimber                 </td><td>0</td></tr>
	<tr><td>male  </td><td>3</td><td>44</td><td>0</td><td>NA </td><td>Cribb, Mr. John Hatfield                    </td><td>0</td></tr>
	<tr><td>male  </td><td>3</td><td>23</td><td>1</td><td>NA </td><td>Asplund, Mr. Johan Charles                  </td><td>0</td></tr>
	<tr><td>male  </td><td>3</td><td>26</td><td>0</td><td>NA </td><td>Bostandyeff, Mr. Guentcho                   </td><td>0</td></tr>
	<tr><td>male  </td><td>2</td><td>27</td><td>0</td><td>NA </td><td>Pulbaum, Mr. Franz                          </td><td>0</td></tr>
	<tr><td>male  </td><td>3</td><td>17</td><td>0</td><td>NA </td><td>Elias, Mr. Joseph Jr                        </td><td>1</td></tr>
	<tr><td>male  </td><td>1</td><td>41</td><td>0</td><td>D21</td><td>Kenyon, Mr. Frederick R                     </td><td>1</td></tr>
	<tr><td>male  </td><td>3</td><td>31</td><td>1</td><td>NA </td><td>Stranden, Mr. Juho                          </td><td>0</td></tr>
	<tr><td>female</td><td>2</td><td>25</td><td>1</td><td>NA </td><td>Shelley, Mrs. William (Imanita Parrish Hall)</td><td>0</td></tr>
	<tr><td>male  </td><td>3</td><td>NA</td><td>0</td><td>NA </td><td>Petroff, Mr. Pastcho ("Pentcho")            </td><td>0</td></tr>
	<tr><td>male  </td><td>3</td><td>NA</td><td>1</td><td>NA </td><td>O'Keefe, Mr. Patrick                        </td><td>0</td></tr>
	<tr><td>male  </td><td>2</td><td> 2</td><td>1</td><td>NA </td><td>Wells, Master. Ralph Lester                 </td><td>1</td></tr>
	<tr><td>male  </td><td>3</td><td>13</td><td>0</td><td>NA </td><td>Asplund, Master. Filip Oscar                </td><td>4</td></tr>
</tbody>
</table>



# Missing values

Number of missing values in the age variable


```
sub_data %>% summarise(num_na=sum(is.na(age)))
```


<table class="dataframe">
<caption>A tibble: 1 × 1</caption>
<thead>
	<tr><th scope=col>num_na</th></tr>
	<tr><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>263</td></tr>
</tbody>
</table>




number of missing values in each variable


```
sub_data %>% purrr::map_df(~sum(is.na(.)))
```


<table class="dataframe">
<caption>A tibble: 1 × 7</caption>
<thead>
	<tr><th scope=col>sex</th><th scope=col>pclass</th><th scope=col>age</th><th scope=col>survived</th><th scope=col>cabin</th><th scope=col>name</th><th scope=col>sibsp</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>0</td><td>0</td><td>263</td><td>0</td><td>1014</td><td>0</td><td>0</td></tr>
</tbody>
</table>




```

nrow(sub_data)
```


1309


drop samples of the variables age and cabin with nas


```

no_nas<-sub_data %>% filter_at(vars(age,cabin),all_vars(!is.na(.)))
nrow(no_nas)
```


272


# Adding New Variables 

Under this section we will use the housing dataset



```

```


```
housing<-readr::read_csv('data/housing.csv',guess_max = 20)
```

    
    [36m--[39m [1m[1mColumn specification[1m[22m [36m------------------------------------------------------------------------------------------------[39m
    cols(
      longitude = [32mcol_double()[39m,
      latitude = [32mcol_double()[39m,
      housing_median_age = [32mcol_double()[39m,
      total_rooms = [32mcol_double()[39m,
      total_bedrooms = [32mcol_double()[39m,
      population = [32mcol_double()[39m,
      households = [32mcol_double()[39m,
      median_income = [32mcol_double()[39m,
      median_house_value = [32mcol_double()[39m,
      ocean_proximity = [31mcol_character()[39m
    )
    
    
    

Under this section we will select only few variables needed to created new variables


```
housing<-housing %>% select(total_rooms,households,total_bedrooms,total_rooms,
        population,households,ocean_proximity,median_income)
```


```

```

# mutate()
mutate() adds new variables at the end of your dataset


```
head(housing 
      
      %>% mutate( rooms_per_household= total_rooms/households,
                  bedrooms_per_room=total_bedrooms/total_rooms,
                  population_per_household=population/households
                  ) 
                  %>% select(-c(total_rooms,households,population,total_bedrooms,median_income))
                  ,3)
```


<table class="dataframe">
<caption>A tibble: 3 × 4</caption>
<thead>
	<tr><th scope=col>ocean_proximity</th><th scope=col>rooms_per_household</th><th scope=col>bedrooms_per_room</th><th scope=col>population_per_household</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>NEAR BAY</td><td>6.984127</td><td>0.1465909</td><td>2.555556</td></tr>
	<tr><td>NEAR BAY</td><td>6.238137</td><td>0.1557966</td><td>2.109842</td></tr>
	<tr><td>NEAR BAY</td><td>8.288136</td><td>0.1295160</td><td>2.802260</td></tr>
</tbody>
</table>



transmute() only keep the new variables created 



```
head(housing %>% 
            transmute(
                  rooms_per_household= total_rooms/households,
                  bedrooms_per_room=total_bedrooms/total_rooms,
                  population_per_household=population/households
                   ),
                   3)
```


<table class="dataframe">
<caption>A tibble: 3 × 3</caption>
<thead>
	<tr><th scope=col>rooms_per_household</th><th scope=col>bedrooms_per_room</th><th scope=col>population_per_household</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>6.984127</td><td>0.1465909</td><td>2.555556</td></tr>
	<tr><td>6.238137</td><td>0.1557966</td><td>2.109842</td></tr>
	<tr><td>8.288136</td><td>0.1295160</td><td>2.802260</td></tr>
</tbody>
</table>



# Summary Statistics 


```
getmode <- function(n) {
   uniqn <- unique(n)
   uniqn[which.max(tabulate(match(n, uniqn)))]
}
housing %>% summarise(count=n(),mean_income=mean(median_income,na.rm=TRUE),
                      mode_income=getmode(median_income))
```


<table class="dataframe">
<caption>A tibble: 1 × 3</caption>
<thead>
	<tr><th scope=col>count</th><th scope=col>mean_income</th><th scope=col>mode_income</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>20640</td><td>3.870671</td><td>3.125</td></tr>
</tbody>
</table>



Group by one variable

* Note : you can groupe by multiple variables


```
housing %>%group_by(ocean_proximity) %>% summarise(mean_income=mean(median_income,na.rm=TRUE),
                                                   mean_housholde=mean(households))
```


<table class="dataframe">
<caption>A tibble: 5 × 3</caption>
<thead>
	<tr><th scope=col>ocean_proximity</th><th scope=col>mean_income</th><th scope=col>mean_housholde</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>&lt;1H OCEAN </td><td>4.230682</td><td>517.7450</td></tr>
	<tr><td>INLAND    </td><td>3.208996</td><td>477.4476</td></tr>
	<tr><td>ISLAND    </td><td>2.744420</td><td>276.6000</td></tr>
	<tr><td>NEAR BAY  </td><td>4.172885</td><td>488.6162</td></tr>
	<tr><td>NEAR OCEAN</td><td>4.005785</td><td>501.2445</td></tr>
</tbody>
</table>




summary statistics on numerical variables group by ocean proximity


```
housing%>% select(-population)%>%group_by(ocean_proximity) %>%
summarise_if(is.numeric, mean, na.rm = TRUE)
```


<table class="dataframe">
<caption>A tibble: 5 × 5</caption>
<thead>
	<tr><th scope=col>ocean_proximity</th><th scope=col>total_rooms</th><th scope=col>households</th><th scope=col>total_bedrooms</th><th scope=col>median_income</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>&lt;1H OCEAN </td><td>2628.344</td><td>517.7450</td><td>546.5392</td><td>4.230682</td></tr>
	<tr><td>INLAND    </td><td>2717.743</td><td>477.4476</td><td>533.8816</td><td>3.208996</td></tr>
	<tr><td>ISLAND    </td><td>1574.600</td><td>276.6000</td><td>420.4000</td><td>2.744420</td></tr>
	<tr><td>NEAR BAY  </td><td>2493.590</td><td>488.6162</td><td>514.1828</td><td>4.172885</td></tr>
	<tr><td>NEAR OCEAN</td><td>2583.701</td><td>501.2445</td><td>538.6157</td><td>4.005785</td></tr>
</tbody>
</table>




```

```
