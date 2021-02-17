
# Libraries
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)
library(readxl)
library(caret)
# library(xlsx)
# library(rsample)
# library(plyr)
# library(ipred)
# install.packages('e1071', dependencies=TRUE)
# install.packages('caret', dependencies=TRUE)


# change this path to your dataset path
datasetPath <- "C:/Users/Mahdi/Documents/R/classification/Classification.xlsx"  

# sample dataset
data <- as.data.frame(read_excel(datasetPath))


# 
#data %<>% mutate_if(is.factor, as.numeric)



# Partition data to test and train by using last column in dataset
# you must change this section of code for your dataset
test <- data[data$`Train/Test` == 'Test', ]
training <- data[data$`Train/Test` == 'Train', ]
test <- test[ -c(28) ]
training <- training[ -c(28) ]
trainingtarget <- training[27]
testtarget <- test[27]
training<-training[-27]
test<-test[-27]


# The dimnames() command can set column names of a matrix 
data <- as.matrix(data)
dimnames(data) <- NULL
# The dimnames() command can set column names of a matrix 
test <- as.matrix(test)
dimnames(test) <- NULL
# The dimnames() command can set column names of a matrix 
training <- as.matrix(training)
dimnames(training) <- NULL
# The dimnames() command can set column names of a matrix 
trainingtarget <- as.matrix(trainingtarget)
dimnames(trainingtarget) <- NULL
# The dimnames() command can set column names of a matrix 
testtarget <- as.matrix(testtarget)
dimnames(testtarget) <- NULL


# mean Normalize dataset
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- scale(training, center = m, scale = s)
test <- scale(test, center = m, scale = s)

# in train dataset the lable column type is character 
# in this section change char to integer
trainingtarget[trainingtarget == "C0"] <- 0
trainingtarget[trainingtarget == "C1"] <- 1
trainingtarget[trainingtarget == "C2"] <- 2
trainingtarget[trainingtarget == "C3"] <- 3
trainingtarget[trainingtarget == "C4"] <- 4
trainingtarget[trainingtarget == "C5"] <- 5
trainingtarget[trainingtarget == "C6"] <- 6
trainingtarget[trainingtarget == "C7"] <- 7
trainingtarget[trainingtarget == "C8"] <- 8
trainingtarget[trainingtarget == "C9"] <- 9

# in test dataset the lable column type is character 
# in this section change char to integer
testtarget[testtarget == "C0"] <- 0
testtarget[testtarget == "C1"] <- 1
testtarget[testtarget == "C2"] <- 2
testtarget[testtarget == "C3"] <- 3
testtarget[testtarget == "C4"] <- 4
testtarget[testtarget == "C5"] <- 5
testtarget[testtarget == "C6"] <- 6
testtarget[testtarget == "C7"] <- 7
testtarget[testtarget == "C8"] <- 8
testtarget[testtarget == "C9"] <- 9


# change type of all character variable to numeric 
testtarget <- as.numeric(as.character(testtarget))
trainingtarget <- as.numeric(as.character(trainingtarget))

# This function takes a vector or 1 column matrix of class labels and converts it into a matrix with 10 columns, one for each category.
trainingtarget <- to_categorical(y = trainingtarget , num_classes = 10)
testtarget <- to_categorical(y = testtarget , num_classes = 10)



###########################################


# Create Model
model <- keras_model_sequential()


model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(26) , kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%  
  layer_dropout(rate =  0.1) %>% 
  
  layer_dense(units = 8, activation = 'relu' , kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate =  0.1) %>%
  
  layer_dense(units = 8, activation = 'relu' , kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate =  0.1) %>%
  
  layer_dense(units = 10 , activation = 'softmax')   # unit = 10 because we have 10 class in dataset




# Compile
model %>% compile(loss = 'categorical_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = 'accuracy')




# Fit Model
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2 ,
      callbacks = list(
        callback_early_stopping(patience = 10)
      )
    )


# Evaluate
model %>% evaluate(test, testtarget)







pred <- model %>% predict(test)
pred <- as.data.frame(pred)
col <- colnames(pred)[apply(pred,1,which.max)]
col[col == "V1"] <- 'C0'
col[col == "V2"] <- 'C1'
col[col == "V3"] <- 'C2'
col[col == "V4"] <- 'C3'
col[col == "V5"] <- 'C4'
col[col == "V6"] <- 'C5'
col[col == "V7"] <- 'C6'
col[col == "V8"] <- 'C7'
col[col == "V9"] <- 'C8'
col[col == "V10"] <- 'C9'
col <- as.data.frame(col)
testtarget2 <- as.data.frame(testtarget)
col2 <- colnames(testtarget2)[apply(testtarget2,1,which.max)]
col2[col2 == "V1"] <- 'C0'
col2[col2 == "V2"] <- 'C1'
col2[col2 == "V3"] <- 'C2'
col2[col2 == "V4"] <- 'C3'
col2[col2 == "V5"] <- 'C4'
col2[col2 == "V6"] <- 'C5'
col2[col2 == "V7"] <- 'C6'
col2[col2 == "V8"] <- 'C7'
col2[col2 == "V9"] <- 'C8'
col2[col2 == "V10"] <- 'C9'
col2 <- as.data.frame(col2)






# create a matrix with 2 column , one column for predicted value and one column for actual value
predResult <- cbind(col , col2)
predResult <- as.data.frame(predResult)
names(predResult) <- c("Prediction" , "Actual")
predResult$Prediction = factor(predResult$Prediction)
predResult$Actual = factor(predResult$Actual)



#  confusion Matrix
cn <- confusionMatrix(predResult$Prediction , predResult$Actual)

#  accuracy and kappa error and total error
eval <- as.data.frame(cn$overall) ; 
totalError <- 1 - eval$`cn$overall`[1]
KappaError <- 1 - eval$`cn$overall`[2]
accuracy <- eval$`cn$overall`[1]

