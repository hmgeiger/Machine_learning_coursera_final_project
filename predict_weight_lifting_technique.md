# Machine learning to predict good weight lifting form using sensor data

## Heather Geiger: December 14,2017

### Pre-processing data

Read in data.

Data was already downloaded from here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Source of this data:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013

http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har


```r
training <- read.csv("pml-training.csv",header=TRUE,stringsAsFactors=FALSE)
testing <- read.csv("pml-testing.csv",header=TRUE,stringsAsFactors=FALSE)
```

Let's explore the data a bit.

First, how many rows and columns are there in each set?


```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

A lot of columns!
But maybe some of these are not needed.

Going to look through the testing data to get a better sense of it.
This will be quite verbose, so not displaying this part.


```r
testing[,1:20]
testing[,21:40]
testing[,41:60]
testing[,61:80]
testing[,81:100]
testing[,101:120]
testing[,121:140]
testing[,141:160]
```

Looks like there are quite a few columns that are all NA's in the test data.

Let's remove these from both the training and test sets.

Also, the first column is just equal to the row number, so let's remove that as well.


```r
columns_to_remove <- c(1,as.numeric(as.vector(which(apply(testing,2,function(x)length(which(is.na(x) == FALSE))) == 0))))

training <- training[,setdiff(1:ncol(training),columns_to_remove)]
testing <- testing[,setdiff(1:ncol(testing),columns_to_remove)]
```

Are the people in the training and test data sets the same?


```r
table(training$user_name)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     3892     3112     3536     3070     3402     2610
```

```r
table(testing$user_name)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##        1        3        1        4        8        3
```

Yes. Good to know.

When we looked at all columns above, we saw some other columns we might want to remove.

For example, new_window is always "no" in the test data.

Timestamp variables can probably be removed as well.

Finally, the name of num_window implies it may be related to the windows of time convention, but we should look at this in more detail.

How does this variable compare per "classe" variable in the training set?


```r
length(unique(training[,"num_window"]))
```

```
## [1] 858
```

```r
unique_num_window_per_classe <- aggregate(num_window ~ classe,data=training,FUN=function(x)length(unique(x)))
unique_num_window_per_classe 
```

```
##   classe num_window
## 1      A        242
## 2      B        168
## 3      C        151
## 4      D        141
## 5      E        156
```

```r
sum(unique_num_window_per_classe[,2])
```

```
## [1] 858
```

The values for num_window appear to be mutually exclusive for each value in classe.

This suggests it may be some sort of marker for the window rather than giving actual information about the physical lifts.

So, let's remove this variable as well.

Also remove problem_id column from testing. This is the last column.


```r
additional_columns_to_remove <- match(c("raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window"),colnames(training))

training <- training[,setdiff(1:ncol(training),additional_columns_to_remove)]
testing <- testing[,setdiff(1:ncol(testing),additional_columns_to_remove)]

testing <- testing[,1:(ncol(testing) - 1)]
```

Final pre-processing step is just to convert variables to factors as needed.

For each column, check the number of possible values for that column in training.

If very few, examine more and consider converting to factor.

Look at all but the first and last column, since we know we need to convert user_name and classe to factor.


```r
num_unique_values_per_column <- c()

for(i in 2:(ncol(training) - 1))
{
num_unique_values_per_column <- c(num_unique_values_per_column,length(unique(training[,i])))
}

head(num_unique_values_per_column[order(num_unique_values_per_column)])
```

```
## [1]  29  43  66  69  70 140
```

Looks like all the other variables are definitely numeric. So, just convert user_name and classe to factor.


```r
training$user_name <- factor(training$user_name)
training$classe <- factor(training$classe)
testing$user_name <- factor(testing$user_name)
```

### Building a model using machine learning

Now, let's do the actual machine learning part.

First, remove user name.

Though it might improve our results a bit, it could also result in a less generalizable model if we want to apply these results to other people.


```r
training_no_user_name <- training[,2:ncol(training)]
```

Now, let's try starting with a nice simple model, rpart.

We'll do a simple holdout method (70%/30% split) to get a general sense of accuracy.

Then we can run cross-validation if this looks promising.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
set.seed(1392)

#Partition by a combination of user name and classe.
#Even though we are not including user name in model, we don't want any one user overrepresented.

training_indices <- createDataPartition(y = paste0(training$user_name,"-",training$classe),p=0.7,list = FALSE)
testing_indices <- setdiff(1:nrow(training),training_indices)

training_for_machine_learning <- training_no_user_name[training_indices,]
testing_for_machine_learning <- training_no_user_name[testing_indices,]

rpart_model <- train(classe ~ .,data=training_for_machine_learning,method="rpart")
rpart_predictions <- predict(rpart_model,newdata=training_for_machine_learning[,1:(ncol(training_for_machine_learning) - 1)])
confusionMatrix(data=rpart_predictions,reference = training_for_machine_learning$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3561 1124 1117 1010  354
##          B   52  896   78  399  345
##          C  284  641 1204  846  673
##          D    0    0    0    0    0
##          E   11    0    0    0 1156
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4957          
##                  95% CI : (0.4874, 0.5041)
##     No Information Rate : 0.2842          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3409          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9112  0.33672  0.50188    0.000  0.45728
## Specificity            0.6337  0.92119  0.78471    1.000  0.99902
## Pos Pred Value         0.4969  0.50621  0.33004      NaN  0.99057
## Neg Pred Value         0.9473  0.85268  0.88172    0.836  0.89097
## Prevalence             0.2842  0.19351  0.17446    0.164  0.18384
## Detection Rate         0.2590  0.06516  0.08756    0.000  0.08407
## Detection Prevalence   0.5211  0.12872  0.26529    0.000  0.08487
## Balanced Accuracy      0.7725  0.62895  0.64329    0.500  0.72815
```

Accuracy is not good at all. Looks like we will need to use a more complex model, even if takes more time.

Let's use the random forest model.

Default parameters should be fine here. 

Defaults are 500 trees, which seems reasonable, along with mtry = sqrt of variable number ~ 7 which also seems reasonable.

We will run 10-fold cross-validation, so no need to use the holdout method.

We can use all rows in training, just leaving out 1/10 at a time.


```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
permutation_training_rows <- sample(1:nrow(training),replace=FALSE)

starts_kfolds <- 1962*1:10 - 1961
ends_kfolds <- 1962*1:10;ends_kfolds[10] <- 19622

accuracy_per_kfold <- c()

for(i in 1:10)
{
this_k_range <- permutation_training_rows[starts_kfolds[i]:ends_kfolds[i]]
not_this_k <- setdiff(1:nrow(training),this_k_range)
rf_model <- randomForest(classe ~ .,data=training_no_user_name[not_this_k,],keep.forest=TRUE, proximity=FALSE,importance=TRUE,do.trace=100)
rf_predictions <- predict(rf_model,newdata=training_no_user_name[this_k_range,1:(ncol(training_no_user_name) - 1)])
accuracy_per_kfold <- c(accuracy_per_kfold,confusionMatrix(data=rf_predictions,reference = training_no_user_name$classe[this_k_range])$overall[1])
}
```

```
## ntree      OOB      1      2      3      4      5
##   100:   0.39%  0.12%  0.44%  0.65%  0.86%  0.09%
##   200:   0.37%  0.12%  0.47%  0.49%  0.76%  0.18%
##   300:   0.37%  0.12%  0.50%  0.52%  0.73%  0.15%
##   400:   0.36%  0.10%  0.44%  0.49%  0.80%  0.15%
##   500:   0.35%  0.10%  0.44%  0.49%  0.76%  0.15%
## ntree      OOB      1      2      3      4      5
##   100:   0.41%  0.04%  0.56%  0.52%  0.97%  0.22%
##   200:   0.35%  0.06%  0.44%  0.39%  0.83%  0.22%
##   300:   0.33%  0.04%  0.41%  0.39%  0.80%  0.22%
##   400:   0.32%  0.06%  0.44%  0.39%  0.69%  0.22%
##   500:   0.32%  0.04%  0.41%  0.39%  0.76%  0.22%
## ntree      OOB      1      2      3      4      5
##   100:   0.42%  0.08%  0.44%  0.72%  0.93%  0.21%
##   200:   0.40%  0.06%  0.50%  0.69%  0.75%  0.21%
##   300:   0.38%  0.06%  0.50%  0.59%  0.75%  0.21%
##   400:   0.35%  0.06%  0.47%  0.56%  0.72%  0.15%
##   500:   0.35%  0.08%  0.44%  0.56%  0.72%  0.15%
## ntree      OOB      1      2      3      4      5
##   100:   0.42%  0.04%  0.61%  0.59%  0.82%  0.31%
##   200:   0.35%  0.02%  0.50%  0.46%  0.72%  0.25%
##   300:   0.35%  0.00%  0.56%  0.49%  0.69%  0.25%
##   400:   0.33%  0.02%  0.47%  0.46%  0.69%  0.25%
##   500:   0.31%  0.00%  0.35%  0.46%  0.72%  0.22%
## ntree      OOB      1      2      3      4      5
##   100:   0.42%  0.06%  0.61%  0.55%  0.83%  0.28%
##   200:   0.36%  0.06%  0.50%  0.42%  0.80%  0.22%
##   300:   0.33%  0.04%  0.50%  0.45%  0.70%  0.19%
##   400:   0.36%  0.04%  0.44%  0.55%  0.73%  0.25%
##   500:   0.33%  0.02%  0.44%  0.39%  0.76%  0.25%
## ntree      OOB      1      2      3      4      5
##   100:   0.41%  0.06%  0.56%  0.46%  1.00%  0.22%
##   200:   0.37%  0.04%  0.53%  0.40%  0.93%  0.19%
##   300:   0.35%  0.02%  0.50%  0.40%  0.89%  0.15%
##   400:   0.32%  0.04%  0.47%  0.36%  0.79%  0.12%
##   500:   0.32%  0.04%  0.44%  0.36%  0.79%  0.19%
## ntree      OOB      1      2      3      4      5
##   100:   0.40%  0.06%  0.50%  0.55%  0.94%  0.22%
##   200:   0.36%  0.04%  0.44%  0.52%  0.80%  0.22%
##   300:   0.32%  0.02%  0.38%  0.45%  0.80%  0.15%
##   400:   0.32%  0.02%  0.44%  0.39%  0.76%  0.19%
##   500:   0.32%  0.02%  0.44%  0.42%  0.80%  0.12%
## ntree      OOB      1      2      3      4      5
##   100:   0.42%  0.04%  0.58%  0.61%  0.83%  0.31%
##   200:   0.39%  0.04%  0.53%  0.55%  0.69%  0.34%
##   300:   0.35%  0.04%  0.41%  0.61%  0.66%  0.25%
##   400:   0.34%  0.04%  0.41%  0.52%  0.69%  0.25%
##   500:   0.33%  0.04%  0.41%  0.52%  0.73%  0.19%
## ntree      OOB      1      2      3      4      5
##   100:   0.39%  0.12%  0.47%  0.65%  0.70%  0.18%
##   200:   0.37%  0.06%  0.50%  0.55%  0.76%  0.18%
##   300:   0.35%  0.06%  0.47%  0.52%  0.73%  0.18%
##   400:   0.36%  0.06%  0.41%  0.55%  0.76%  0.22%
##   500:   0.32%  0.06%  0.38%  0.49%  0.63%  0.22%
## ntree      OOB      1      2      3      4      5
##   100:   0.44%  0.10%  0.53%  0.68%  0.87%  0.25%
##   200:   0.36%  0.08%  0.38%  0.55%  0.83%  0.18%
##   300:   0.39%  0.08%  0.53%  0.49%  0.87%  0.18%
##   400:   0.36%  0.06%  0.47%  0.49%  0.83%  0.15%
##   500:   0.37%  0.06%  0.50%  0.52%  0.83%  0.18%
```

```r
accuracy_per_kfold
```

```
##  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy  Accuracy 
## 0.9943935 0.9969419 0.9974516 0.9959225 0.9959225 0.9989806 0.9959225 
##  Accuracy  Accuracy  Accuracy 
## 0.9979613 0.9969419 0.9984725
```

Accuracy seems quite good!

Let's make a model including all rows in training now.


```r
rf_model <- randomForest(classe ~ .,data=training_no_user_name,keep.forest=TRUE, proximity=FALSE,importance=TRUE,do.trace=100)
```

```
## ntree      OOB      1      2      3      4      5
##   100:   0.35%  0.05%  0.40%  0.56%  0.78%  0.17%
##   200:   0.29%  0.05%  0.32%  0.38%  0.78%  0.11%
##   300:   0.27%  0.04%  0.26%  0.38%  0.72%  0.14%
##   400:   0.28%  0.04%  0.29%  0.47%  0.65%  0.11%
##   500:   0.28%  0.04%  0.32%  0.44%  0.62%  0.14%
```

Finally, predict for the 20 testing cases we started with, where classe is unknown.


```r
testing_predictions <- predict(rf_model,newdata=testing)
```

### Conclusions

The accuracy on the 10 iterations of our k-fold cross validation was quite good (99.4-99.9%). In theory our out-of-sample error should be relatively similar to the 0.1-0.6% we see with the cross validation, which would be quite good.

Based on these accuracy rates, there is a pretty good chance that we will get all 20 instances right in the testing set.

It would be interesting to test this model on a set based on users besides the 6 included in this initial study.

Even though we excluded user name in the model, other users might have substantially different movement patterns than these 6 individuals, which could reduce the predictive power of the model and increase our out-of-sample error.
