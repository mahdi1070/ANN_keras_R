# Libraries
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)
library(readxl)
library(caret)
# library(xlsx)
library(rsample)
# library(reticulate)
# sys <- import("sys", convert = TRUE)
# sys$path

# change this path to your dataset path
datasetPath <- "C:/Users/Mahdi/Documents/R/classification/Regression.xlsx"  

# sample dataset
data <- as.data.frame(read_excel(datasetPath))


# data %<>% mutate_if(is.factor, as.numeric)




# Matrix
data <- as.matrix(data)
dimnames(data) <- NULL




# Partition data to train and test
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(.7, .3))
training <- data[ind==1,1:4]
test <- data[ind==2, 1:4]
trainingtarget <- data[ind==1, 5]
testtarget <- data[ind==2, 5]



# Normalize
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)
test <- scale(test, center = m, scale = s)



# just for regression
# class column log mishe --- scale class
trainingtarget <- log(trainingtarget)
testtarget <- log(testtarget)





###########################################


# Create Model
model <- keras_model_sequential()


model %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = c(4) , kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate =  0.1) %>%
  
  layer_dense(units = 16, activation = 'relu' , kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate =  0.1) %>%
  
  layer_dense(units = 16, activation = 'relu' , kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate =  0.1) %>%
  
  layer_dense(units = 1)




# Compile
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')



# Fit Model
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 100,
      batch_size = 32,  
      validation_split = 0.2,
      callbacks = list(
        callback_early_stopping(patience = 10)
      )
  )




# Evaluate
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)




preds1 <- as.data.frame(exp(pred))
actual <- as.data.frame(exp(testtarget))
names(actual) <- c('V1')
rmse <- RMSE(preds1$V1,actual$V1) ## RMSE


rss <- sum((preds1$V1 - actual$V1) ^ 2)  ## residual sum of squares
tss <- sum((actual$V1 - mean(actual$V1)) ^ 2)  ## total sum of squares
rsq <- 1 - rss/tss  ## R square (r^2)


