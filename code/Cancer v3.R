#############################################################################################
###################################   CANCER ANALYSIS #######################################
#############################################################################################


##############################################
###########  SET-UP ENVIRONMENT ##############
##############################################

##CLEAR WORK ENVIRONMENT
rm(list = ls())


##LOAD PACKAGES
##library("caret")
##library("rpart")
##library("rattle")
##library("sampling")
##library("mlbench")
if(!require("caret")) install.packages("caret"); library("caret")
if(!require("rpart")) install.packages("rpart"); library("rpart")
if(!require("rattle")) install.packages("rattle"); library("rattle")
if(!require("sampling")) install.packages("sampling"); library("sampling")
if(!require("mlbench")) install.packages("mlbench"); library("mlbench")

##SET SEED
set.seed(100)


##GET FILE
file <- "./data/Wisconsin_Breast_Cancer_Dataset.csv"
data <- read.csv(file,  header = TRUE)






##############################################
#########  EXPLORATORY ANALYSIS ##############
##############################################

##EXPLORATORY REPORTS
summary(data)  
str(data)
table(data$diagnosis) 
prop.table(table(data$diagnosis))







##############################################
#########  RFE FEATURE SELECTION #############
##############################################

##DEFINE IMPORTANT FEATURES
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
results <- rfe(data[,-1], data[,1], rfeControl=control)
print(results)
plot(results, type=c("g", "o"))

##SELECT IMPORTANT FEATURES
predictors <- predictors(results)
predictors <- c(predictors, "diagnosis")
data <- data[,predictors]
str(data)





##############################################
#########  UPDATE ENVIRONMENT  ##############
##############################################

##CREATE TRAIN AND TEST DATA 
train.index <- createDataPartition(data$diagnosis, p = .7, list = FALSE)

data.train <- data[train.index,]
table(data.train$diagnosis)
prop.table(table(data.train$diagnosis))

data.test <- data[-train.index,]
table(data.test$diagnosis)
prop.table(table(data.test$diagnosis))




##############################################
############  KNN MODEL ANALYSIS #############
##############################################
##SET SEED
set.seed(842)


##RELABEL DATA
data.training = data.train 
data.testing = data.test


##CREATE PERFORMANCE CURVE
set.seed(84654333)
ctrl <- trainControl(method="cv",number = 10) 
tuneGrid = expand.grid(k = c(1,2,3,4,5,10,20,30,40,50,100,150)) ##modelLookup("knn") ##names(getModelInfo())
(model.knn <- train(diagnosis ~ ., data = data.training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid))
plot(model.knn)
plot(model.knn$results$k, model.knn$results$Accuracy, type = "b", xlab = "K-Neighbors", ylab = "Accuracy", main = "Performance Curve")

##TEST MODEL
predict.knn <- predict(model.knn, newdata = data.testing)
confusionMatrix(predict.knn, data.testing$diagnosis)


##CREATE LEARNING CURVE 
set.seed(874521)
k.optimal = model.knn$bestTune$k
tuneGrid = expand.grid(k = k.optimal)

size.percent <- seq(from = .10, to = 1, by = .05)
size.num <- NULL
kappa.cv <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.train <- NULL

table(data.training$diagnosis)
for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$diagnosis == "B",]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$diagnosis == "M",]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("diagnosis"), size = c(size1,size0) , method = "srswor")
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$diagnosis))
  
  ##GET CV ERROR
  ctrl <- trainControl(method="cv",number = 10) 
  model.knn <- train(diagnosis ~ ., data = data.strata, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid)
  kappa.cv <- c(kappa.cv,  model.knn$results$Kappa )
  accuracy.cv <- c(accuracy.cv, model.knn$results$Accuracy)
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  model.knn <- train(diagnosis ~ ., data = data.strata, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid)
  predict.knn <- predict(model.knn,newdata = data.strata)
  cf.knn <- confusionMatrix(predict.knn, data.strata$diagnosis )
  kappa.train <- c(kappa.train, cf.knn$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.knn$overall[["Accuracy"]])
  
}


##NEED TO PUT ON SAME CHART
plot(size.num, accuracy.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Accuracy", main = "Learning Curve" )
points(size.num, accuracy.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Training","Cross Validation"), fill=c("red","black"), horiz=FALSE)









##############################################
############ DECISION TREE MODEL #############
##############################################

##SET SEED
set.seed(842)

##RELABEL DATA
data.training = data.train 
data.testing = data.test


##CREATE PERFORMANCE CURVE
set.seed(71209)
ctrl <- trainControl(method="cv",number = 10) 
ctrl.rpart <- rpart.control(minsplit = 2, minibucket = 1, cp = 0)
maxdepth <- seq(from = 1, t= 30, by = 1)
tuneGrid = expand.grid(maxdepth = maxdepth) ##modelLookup("knn") ##names(getModelInfo())
method <- "rpart2"

(model.tree.gini <- train(diagnosis ~ ., data = data.training, method = method, 
                     trControl = ctrl, control = ctrl.rpart, 
                     tuneGrid = tuneGrid, parms = list(split="gini")))

(model.tree.information <- train(diagnosis ~ ., data = data.training, method = method, 
                          trControl = ctrl, control = ctrl.rpart, 
                          tuneGrid = tuneGrid, parms = list(split="information")))


plot(model.tree.gini$results$maxdepth, model.tree.gini$results$Accuracy, type = "b", 
     col = "red", ylim=c(.50,1), xlab = "(Max) Tree Depth", ylab = "Accuracy", main = "Performance Curve")
points(model.tree.information$results$Accuracy, type = "b", col = "black")
legend("bottomright", inset=.01, title="Legend",
       c("Gini Index","Information Gain"), fill=c("red","black"), horiz=FALSE)


##TEST MODEL
predict.tree <- predict(model.tree.information, newdata = data.testing)
confusionMatrix(predict.tree, data.testing$diagnosis)






##CREATE LEARNING CURVE 
set.seed(83201)
maxdepth.optimal = model.tree.information$bestTune$maxdepth
##maxdepth.optimal = 1  
tuneGrid <- expand.grid(maxdepth = maxdepth.optimal)
method = "rpart2"

size.percent <- c(.10, .20, .30, .40, .50, .60, .70, .80, .90, 1)
size.num <- NULL
kappa.cv <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.train <- NULL

table(data.training$diagnosis)
for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$diagnosis == "B",]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$diagnosis == "M",]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("diagnosis"), size = c(size1,size0) , method = "srswor")
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$diagnosis))
  
  ##GET CV ERROR
  ctrl <- trainControl(method="cv",number = 10)
  model.tree <- train(diagnosis ~ ., data = data.strata, method = method, trControl = ctrl, tuneGrid = tuneGrid)
  kappa.cv <- c(kappa.cv,  model.tree$results$Kappa )
  accuracy.cv <- c(accuracy.cv, model.tree$results$Accuracy)
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  model.tree <- train(diagnosis ~ ., data = data.strata, method = method, trControl = ctrl, tuneGrid = tuneGrid)
  predict.tree <- predict(model.tree,newdata = data.strata)
  cf.tree <- confusionMatrix(predict.tree, data.strata$diagnosis )
  kappa.train <- c(kappa.train, cf.tree$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.tree$overall[["Accuracy"]])
  
  
}


##NEED TO PUT ON SAME CHART
plot(size.num, accuracy.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Accuracy", main = "Learning Curve" )
points(size.num, accuracy.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Training","Cross Validation"), fill=c("red","black"), horiz=FALSE)









##############################################
############ SVM MODEL ANALYSIS ##############
##############################################

##SET SEED
set.seed(94318) 

##RELABEL DATA
data.training = data.train 
data.testing = data.test


##CREATE PERFORMANCE CURVE - LINEAR KERNEL
set.seed(94318)
ctrl <- trainControl(method="cv",number = 10) 
cost <- c(0.01, 0.25, 0.50, 0.75, 1,2)
tuneGrid = expand.grid(cost = cost) ##modelLookup("svmLinear2") ##names(getModelInfo()) 
(model.svm.linear <- train(diagnosis ~ ., data = data.training, method = "svmLinear2", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid))
plot(model.svm.linear$results$cost, model.svm.linear$results$Accuracy, type = "b", xlab = "Cost", 
     ylab = "Accuracy", main = "SVM (Linear Kernel) Performance Curve" )


##TEST MODEL - LINEAR KERNEL 
pred <- predict(model.svm.linear, data.testing)
(cf <- confusionMatrix(pred, data.testing$diagnosis))



##CREATE PERFORMANCE CURVE - RADIAL KERNEL
set.seed(943018)
ctrl <- trainControl(method="cv",number = 5)
sigma <-c(0.01, 0.1, 1,10)
C <- c(0.01, 0.1, 1,10)
tuneGrid = expand.grid( sigma = sigma, C = C) ##modelLookup("svmLinear2") ##names(getModelInfo()) 
(model.svm.radial <- train(diagnosis ~ ., data = data.training, method = "svmRadial", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid))
plot(model.svm.radial, xlab = "Cost", ylab = "Kappa", main = "SVM (Radial Kernel) Performance Curve for Sigma")

##TEST MODEL - RADIAL KERNEL 
pred <- predict(model.svm.radial, data.testing)
(cf <- confusionMatrix(pred, data.testing$diagnosis))




##CREATE PERFORMANCE CURVE - RADIAL KERNEL (SIGMA)
set.seed(91260)
ctrl <- trainControl(method="cv",number = 10) 
C <-  1
sigma <- c(seq(from = 0.2, to = 1, by = 0.2), seq (from = 2, to = 5, by = 1))
tuneGrid = expand.grid( sigma = sigma, C = C) ##modelLookup("svmLinear2") ##names(getModelInfo())
(model.svm.radial <- train(diagnosis ~ ., data = data.training, method = "svmRadial", 
                           trControl = ctrl, preProcess = c("center","scale"), 
                           tuneGrid = tuneGrid, metric = "Accuracy"))
plot(model.svm.radial,xlab = "Sigma", ylab = "Accuracy", main = "SVM (Radial Kernel) Performance Curve for Sigma")

##TEST MODEL - RADIAL KERNEL 
pred <- predict(model.svm.radial, data.testing)
(cf <- confusionMatrix(pred, data.testing$diagnosis))



##PLOT LEARNING CURVE
set.seed(9000)
C <- model.svm.radial$bestTune$C
##C <- .5
sigma <- model.svm.radial$bestTune$sigma
tuneGrid = expand.grid(sigma = sigma, C = C)  ##modelLookup("svmRadial") ##names(getModelInfo())
method = "svmRadial"


size.percent <- seq(from = .1, to = 1, by = .1)
size.num <- NULL
kappa.cv <- NULL
kappa.cv.model <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.cv.model <- NULL
accuracy.train <- NULL


for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$diagnosis == "B",]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$diagnosis == "M",]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("diagnosis"), size = c(size1,size0) , method = "srswor")  
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$diagnosis))
  
  ##GET CV ERROR
  ctrl <- trainControl(method="cv",number = 10) 
  (model <- train(diagnosis ~ ., data.strata, method = method, verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
  
  kappa.cv <- c(kappa.cv, model$results[["Kappa"]])
  accuracy.cv <- c(accuracy.cv, model$results[["Accuracy"]])
  
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  (model.train <- train(diagnosis ~ ., data.strata, method = method, verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
  
  pred.train <- predict(model.train, data.strata, type = "raw")
  cf.train <- confusionMatrix(pred.train, data.strata$diagnosis)
  kappa.train <- c(kappa.train, cf.train$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.train$overall[["Accuracy"]])
  
}

##NEED TO PUT ON SAME CHART
plot(size.num, accuracy.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Accuracy", main = "Learning Curve" )
points(size.num, accuracy.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)









##############################################
############ BOOSTING MODEL ANALYSIS #########
##############################################

##SET SEED
set.seed(17002)

##RELABEL DATA
data.training = data.train 
data.testing = data.test

##CREATE PERFORMANCE CURVE
set.seed(19387)
ctrl <- trainControl(method="cv",number = 10) 
n.trees <- seq(from = 1, to = 75, by = 5)
interaction.depth <- c(1)
shrinkage <- seq(from = .01, to = 0.05, by = 0.02)
n.minobsinnode <- c(20)

tuneGrid = expand.grid(n.trees = n.trees, interaction.depth = interaction.depth, shrinkage = shrinkage, n.minobsinnode = n.minobsinnode) ##modelLookup("gbm") ##names(getModelInfo())
(model <- train(diagnosis ~ ., data.training, method = "gbm", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Accuracy"))
plot(model, ylab = "Accuracy", main = "Performance Curve")

##TEST MODEL 
pred <- predict(model, data.testing)
(cf <- confusionMatrix(pred, data.testing$diagnosis))


##PLOT LEARNING CURVE
set.seed(8011126)
n.trees <- model$bestTune$n.trees
shrinkage <- model$bestTune$shrinkage
interaction.depth <- model$bestTune$interaction.depth
n.minobsinnode <- model$bestTune$n.minobsinnode
tuneGrid = expand.grid(n.trees = n.trees, interaction.depth = interaction.depth, shrinkage = shrinkage, n.minobsinnode = n.minobsinnode) ##modelLookup("gbm") ##names(getModelInfo())


size.percent <- seq(from = .2, to = 1, by = .1)
size.num <- NULL
kappa.cv <- NULL
kappa.cv.model <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.cv.model <- NULL
accuracy.train <- NULL


time.train <- NULL
time.cv <- NULL

for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$diagnosis == "B",]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$diagnosis == "M",]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("diagnosis"), size = c(size1,size0) , method = "srswor")
   
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$diagnosis))
  
  ##GET CV ERROR
  ctrl <- trainControl(method="cv",number = 10) 
  
  (model <- train(diagnosis ~ ., data.strata, method = "gbm", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa", bag.fraction = .75))
  time.cv <- c(time.cv, (model$time$everything["user.self"]))
  
  kappa.cv <- c(kappa.cv, model$results[["Kappa"]])
  accuracy.cv <- c(accuracy.cv, model$results[["Accuracy"]])
  
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  tuneGrid = expand.grid(n.trees = n.trees, interaction.depth = interaction.depth, shrinkage = shrinkage, n.minobsinnode = n.minobsinnode) ##modelLookup("gbm") ##names(getModelInfo())
  

  (model.train <- train(diagnosis ~ ., data.strata, method = "gbm", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
  time.train <- c(time.train, (model.train$time$everything["user.self"]))
  
  pred.train <- predict(model.train, data.strata, type = "raw")
  cf.train <- confusionMatrix(pred.train, data.strata$diagnosis)
  kappa.train <- c(kappa.train, cf.train$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.train$overall[["Accuracy"]])
  
}


##NEED TO PUT ON SAME CHART
plot(size.num, accuracy.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Accuracy", main = "Learning Curve" )
points(size.num, accuracy.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)


##PLOT TIME CURVES
plot(size.num,  time.train, type = "b", col = "red", ylim = c(0,5), 
     ylab = "Seconds", xlab = "Data Size", main = "Time vs Data Size")
points(size.num, time.cv, type = "b", col = "black" )
legend("topleft", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)





##############################################
###############  NEURAL NET  #################
##############################################

##SET SEED
set.seed(74677)

##RELABEL DATA
data.training = data.train 
data.testing = data.test

##CREATE PERFORMANCE CURVE
ctrl <- trainControl(method="cv",number = 10) 
size <- c(1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80)
decay <- 0
tuneGrid = expand.grid(size = size, decay = decay) ##modelLookup("gbm") ##names(getModelInfo())
(model <- train(diagnosis ~ ., data.training, method = "nnet", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Accuracy", maxit = 300))
plot(model, ylab = "Accuracy", xlab = "Hidden Units", main = "Performance Curve")


##TEST MODEL 
pred <- predict(model, data.testing)
(cf <- confusionMatrix(pred, data.testing$diagnosis))


##PLOT LEARNING CURVE
set.seed(106565)
size <- model$bestTune$size
decay <- model$bestTune$decay
tuneGrid = expand.grid(size = size, decay = decay) ##modelLookup("gbm") ##names(getModelInfo())
model.type <- "nnet"

size.percent <- seq(from = .2, to = 1, by = .2)
size.num <- NULL
kappa.cv <- NULL
kappa.cv.model <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.cv.model <- NULL
accuracy.train <- NULL

time.train <- NULL
time.cv <- NULL



for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$diagnosis == "B",]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$diagnosis == "M",]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("diagnosis"), size = c(size1,size0) , method = "srswor")
   
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$diagnosis))
  
  ##GET CV ERROR  **GET CV FROM MODEL**
  ctrl <- trainControl(method="cv",number = 5) 
  (model <- train(diagnosis ~ ., data.strata, method = model.type, verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Accuracy", maxit = 300))
  time.cv <- c(time.cv, (model$time$everything["user.self"]))
  
  kappa.cv <- c(kappa.cv, model$results[["Kappa"]])
  accuracy.cv <- c(accuracy.cv, model$results[["Accuracy"]])
  
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
   (model.train <- train(diagnosis ~ ., data.strata, method = model.type, verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Accuracy", maxit = 300))
  time.train <- c(time.train, (model.train$time$everything["user.self"]))
  
  pred.train <- predict(model.train, data.strata, type = "raw")
  cf.train <- confusionMatrix(pred.train, data.strata$diagnosis)
  kappa.train <- c(kappa.train, cf.train$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.train$overall[["Accuracy"]])
  
}



##NEED TO PUT ON SAME CHART
plot(size.num, accuracy.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Accuracy", main = "Learning Curve" )
points(size.num, accuracy.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)


##PLOT TIME CURVES
plot(size.num,  time.train, type = "b", col = "red", ylim = c(0,20),##ylim = c(0,0.25), 
     ylab = "Seconds", xlab = "Data Size", main = "Time vs Data Size")
points(size.num, time.cv, type = "b", col = "black" )
legend("topleft", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)







