#############################################################################################
###################################   HR ANALYSIS ###########################################
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
if(!require("caret")) install.packages("caret"); library("caret")
if(!require("rpart")) install.packages("rpart"); library("rpart")
if(!require("rattle")) install.packages("rattle"); library("rattle")
if(!require("sampling")) install.packages("sampling"); library("sampling")




##SET SEED
set.seed(100)


##GET FILE
file <- "./data/HR_Analytics_Dataset.csv"
data <- read.csv(file,  header = TRUE)


##CREATE AND RELEVEL FACTORS
data$left <- as.factor(data$left)
data$salary <- factor(data$salary, c("low","medium","high"))
head(data$salary)



##CREATE TRAIN AND TEST DATA 
train.index <- createDataPartition(data$left, p = .7, list = FALSE)

data.train <- data[train.index,]
table(data.train$left)
prop.table(table(data.train$left))

data.test <- data[-train.index,]
table(data.test$left)
prop.table(table(data.test$left))




##############################################
#########  EXPLORATORY ANALYSIS ##############
##############################################

##EXPLORATORY REPORTS
summary(data)  
str(data)
table(data$left) 
prop.table(table(data$left))









##############################################
############  KNN MODEL ANALYSIS #############
##############################################
##SET SEED
set.seed(842)

##ENCODE FACTORS
features <- dummyVars(" ~ department", data = data.train)
features.encoded <- data.frame(predict(features, newdata = data.train))
data.train.encoded <- cbind(data.train[,!(colnames(data.train) %in% c("department"))],features.encoded)
str(data.train.encoded)

features <- dummyVars(" ~ department", data = data.test)
features.encoded <- data.frame(predict(features, newdata = data.test))
data.test.encoded <- cbind(data.test[,!(colnames(data.test) %in% c("department"))],features.encoded)
str(data.test.encoded)


##RELABEL DATA
data.training = data.train.encoded 
data.testing = data.test.encoded


##CREATE PERFORMANCE CURVE
set.seed(10876)
ctrl <- trainControl(method="cv",number = 10) 
tuneGrid = expand.grid(k = c(1,2,3,4,5,10,20,30,40,50,100, 150)) ##modelLookup("knn") ##names(getModelInfo())
(model.knn <- train(left ~ ., data = data.training, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid))
plot(model.knn)
plot(model.knn$results$k, model.knn$results$Kappa, type = "b", xlab = "K-Neighbors", ylab = "Kappa", main = "Performance Curve")

##TEST MODEL
predict.knn <- predict(model.knn, newdata = data.testing)
confusionMatrix(predict.knn, data.testing$left)


##CREATE LEARNING CURVE 
set.seed(98541)
k.optimal =  model.knn$bestTune$k ## 1 ##FIX THIS
size.percent <- seq(from = 0.05, to = 1, by = 0.05)
size.num <- NULL
kappa.cv <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.train <- NULL

table(data.training$left)
for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$left == 0,]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$left == 1,]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("left"), size = c(size1,size0) , method = "srswor")
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$left))
  
  ##GET CV ERROR
  ctrl <- trainControl(method="cv",number = 10) 
  tuneGrid = expand.grid(k = k.optimal)
  model.knn <- train(left ~ ., data = data.strata, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid)
  kappa.cv <- c(kappa.cv,  model.knn$results$Kappa )
  accuracy.cv <- c(accuracy.cv, model.knn$results$Accuracy)
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  tuneGrid = expand.grid(k = k.optimal)
  model.knn <- train(left ~ ., data = data.strata, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneGrid = tuneGrid)
  predict.knn <- predict(model.knn,newdata = data.strata)
  cf.knn <- confusionMatrix(predict.knn, data.strata$left )
  kappa.train <- c(kappa.train, cf.knn$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.knn$overall[["Accuracy"]])
  
  
}

##NEED TO PUT ON SAME CHART
plot(size.num, kappa.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Kappa", main = "Learning Curve" )
points(size.num, kappa.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Training","Cross Validation"), fill=c("red","black"), horiz=FALSE)


  







##############################################
############ DECISION TREE MODEL #############
##############################################

##RELABEL DATA
data.training = data.train 
data.testing = data.test


##CREATE PERFORMANCE CURVE
set.seed(187543)
ctrl <- trainControl(method="cv",number = 10) 
ctrl.rpart <- rpart.control(minsplit = 2, minibucket = 1, cp = 0)
ctrl.rpart <- NULL
maxdepth <- seq(from = 1, to = 30, by = 1)
tuneGrid = expand.grid(maxdepth = maxdepth) ##modelLookup("knn") ##names(getModelInfo())
method <- "rpart2"


(model.tree.gini <- train(left ~ ., data = data.training, method = method, metric = "Kappa", 
                     trControl = ctrl, control = ctrl.rpart, tuneGrid = tuneGrid,  
                     parms=list(split='gini') ))

(model.tree.information <- train(left ~ ., data = data.training, method = method, metric = "Kappa", 
                     trControl = ctrl, control = ctrl.rpart, tuneGrid = tuneGrid,  
                     parms=list(split='information') ))



plot(model.tree.gini$results$maxdepth, model.tree.gini$results$Kappa, type = "b", col = "red",  
     xlab = "(Max) Tree Depth", ylab = "Kappa", main = "Performance Curve")
points(model.tree.information$results$Kappa, type = "b", col = "black")
legend("bottomright", inset=.01, title="Legend",
       c("Gini Index","Information Gain"), fill=c("red","black"), horiz=FALSE)



##TEST MODEL
predict.tree <- predict(model.tree.information, newdata = data.testing)
confusionMatrix(predict.tree, data.testing$left)



##CREATE LEARNING CURVE 
set.seed(18611)
maxdepth.optimal = model.tree.information$bestTune$maxdepth
tuneGrid <- expand.grid(maxdepth = maxdepth.optimal)
method = "rpart2"


size.percent <- seq(from = .05, to = 1, by = .05)
size.num <- NULL
kappa.cv <- NULL
kappa.train <- NULL
accuracy.cv <- NULL
accuracy.train <- NULL

table(data.training$left)
for( i in 1:length(size.percent))
{
  ##STRATIFY DATA
  size.train = size.percent[i]
  size0 = round(nrow(data.training[data.training$left == 0,]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$left == 1,]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("left"), size = c(size1,size0) , method = "srswor")
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$left))
  
  ##GET CV ERROR
  ctrl <- trainControl(method="cv",number = 10)
  model.tree <- train(left ~ ., data = data.strata, method = method, trControl = ctrl, tuneGrid = tuneGrid)
  kappa.cv <- c(kappa.cv,  model.tree$results$Kappa )
  accuracy.cv <- c(accuracy.cv, model.tree$results$Accuracy)
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  model.tree <- train(left ~ ., data = data.strata, method = method, trControl = ctrl, tuneGrid = tuneGrid)
  predict.tree <- predict(model.tree,newdata = data.strata)
  cf.tree <- confusionMatrix(predict.tree, data.strata$left )
  kappa.train <- c(kappa.train, cf.tree$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.tree$overall[["Accuracy"]])
  
  
}


##PLOT CURVES
plot(size.num, kappa.cv, type = "b", col = "black",  ylim=c(.40,1), xlab = "Data Size", ylab = "Kappa", main = "Learning Curve" )
points(size.num, kappa.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)












##############################################
############ SVM MODEL ANALYSIS ##############
##############################################

##SET SEED
set.seed(94318)


##ENCODE FACTORS
features <- dummyVars(" ~ department", data = data.train)
features.encoded <- data.frame(predict(features, newdata = data.train))
data.train.encoded <- cbind(data.train[,!(colnames(data.train) %in% c("department"))],features.encoded)
str(data.train.encoded)

features <- dummyVars(" ~ department", data = data.test)
features.encoded <- data.frame(predict(features, newdata = data.test))
data.test.encoded <- cbind(data.test[,!(colnames(data.test) %in% c("department"))],features.encoded)
str(data.test.encoded)


##RELABEL DATA
data.training = data.train.encoded 
data.testing = data.test.encoded


##CREATE PERFORMANCE CURVE - LINEAR KERNEL
set.seed(7654866)
ctrl <- trainControl(method="cv",number = 10) 
cost = c(0.01, 0.25, 0.50, 0.75, 1)
tuneGrid = expand.grid(cost = cost) ##Cost defines the tradeoff between error and margin ##modelLookup("svmLinear2") ##names(getModelInfo()) ##svmLinear2 = e1071
(model.svm.linear <- train(left ~ ., data = data.training, method = "svmLinear2", 
                           trControl = ctrl, preProcess = c("center","scale"), 
                           tuneGrid = tuneGrid, metric = "Kappa"))
plot(model.svm.linear)
plot(model.svm.linear$results$cost, model.svm.linear$results$Kappa, type = "b", xlab = "Cost", 
     ylab = "Kappa", main = "SVM (Linear Kernel) Performance Curve" )

##TEST MODEL - LINEAR KERNEL 
pred <- predict(model.svm.linear, data.testing)
(cf <- confusionMatrix(pred, data.testing$left))




##CREATE PERFORMANCE CURVE - RADIAL KERNEL (COST + SIGMA)
set.seed(91260)
ctrl <- trainControl(method="cv",number = 5) 
sigma <- c(0.01, 0.1, 1,10)
C <- c(0.01, 0.1, 1,2)


##tuneGrid = expand.grid( sigma = c(0.01, 0.1, 1,10), C = c(0.01, 0.1, 1,10)) ##Cost defines the tradeoff between error and margin ##modelLookup("svmLinear2") ##names(getModelInfo()) ##svmLinear2 = e1071
tuneGrid = expand.grid( sigma = sigma, C = C) ##Cost defines the tradeoff between error and margin ##modelLookup("svmLinear2") ##names(getModelInfo()) ##svmLinear2 = e1071

(model.svm.radial <- train(left ~ ., data = data.training, method = "svmRadial", 
                           trControl = ctrl, preProcess = c("center","scale"), 
                           tuneGrid = tuneGrid, metric = "Kappa"))
##PLOT
plot(model.svm.radial, xlab = "Cost", ylab = "Kappa", main = "SVM (Radial Kernel) Performance Curve for Sigma and Cost")


##TEST MODEL - RADIAL KERNEL 
pred <- predict(model.svm.radial, data.testing)
(cf <- confusionMatrix(pred, data.testing$left))


##CREATE PERFORMANCE CURVE - RADIAL KERNEL (SIGMA)
set.seed(91260)
ctrl <- trainControl(method="cv",number = 5) 
C <-  1
sigma <- c(seq(from = 0.2, to = 1, by = 0.2), seq (from = 2, to = 5, by = 1))
tuneGrid = expand.grid( sigma = sigma, C = C) ##Cost defines the tradeoff between error and margin ##modelLookup("svmLinear2") ##names(getModelInfo()) ##svmLinear2 = e1071
(model.svm.radial <- train(left ~ ., data = data.training, method = "svmRadial", 
                           trControl = ctrl, preProcess = c("center","scale"), 
                           tuneGrid = tuneGrid, metric = "Kappa"))
plot(model.svm.radial,xlab = "Sigma", ylab = "Kappa", main = "SVM (Radial Kernel) Performance Curve for Sigma")

##TEST MODEL - RADIAL KERNEL 
pred <- predict(model.svm.radial, data.testing)
(cf <- confusionMatrix(pred, data.testing$left))


##PLOT LEARNING CURVE
set.seed(2467890)
C <- model.svm.radial$bestTune$C
sigma <- model.svm.radial$bestTune$sigma
tuneGrid = expand.grid(sigma = sigma, C = C)  ##modelLookup("svmRadial") ##names(getModelInfo())

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
  size0 = round(nrow(data.training[data.training$left == 0,]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$left == 1,]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("left"), size = c(size1,size0) , method = "srswor")  
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$left))
  
  ##GET CV ERROR 
  ctrl <- trainControl(method="cv",number = 10) 
  (model <- train(left ~ ., data.strata, method = "svmRadial", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa", bag.fraction = 1)) ##BAG FRACTION DOES NOT MAKE SENSE HERE
  
  kappa.cv.model <- c(kappa.cv.model, model$results[["Kappa"]])
  accuracy.cv.model <- c(accuracy.cv.model, model$results[["Accuracy"]])
  
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  (model.train <- train(left ~ ., data.strata, method = "svmRadial", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
  
  pred.train <- predict(model.train, data.strata, type = "raw")
  cf.train <- confusionMatrix(pred.train, data.strata$left)
  kappa.train <- c(kappa.train, cf.train$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.train$overall[["Accuracy"]])
  
}



##PLOT CURVES
plot(size.num, kappa.cv.model, type = "b", col = "black",  ylim=c(0,1), xlab = "Data Size", ylab = "Kappa", main = "Learning Curve" )
points(size.num, kappa.train, type = "b", col = "red")
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
set.seed(7866)
ctrl <- trainControl(method="cv",number = 10) 
tuneGrid = expand.grid(n.trees = c(50,100,150,200,250, 300, 350), interaction.depth = c(1), shrinkage = c(0.01, 0.1, 1), n.minobsinnode = c(100)) ##modelLookup("gbm") ##names(getModelInfo())
(model <- train(left ~ ., data.training, method = "gbm", verbose = FALSE, trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
plot(model, main = "Performance Curve", ylab = "Kappa")

##TEST MODEL 
pred <- predict(model, data.testing)
(cf <- confusionMatrix(pred, data.testing$left))


##PLOT LEARNING CURVE
set.seed(132211)
n.trees <- model$bestTune$n.trees
shrinkage <- model$bestTune$shrinkage
interaction.depth <- model$bestTune$interaction.depth
n.minobsinnode <- 100
tuneGrid = expand.grid(n.trees = n.trees, interaction.depth = interaction.depth, 
                       shrinkage = shrinkage, n.minobsinnode = n.minobsinnode) ##modelLookup("gbm") ##names(getModelInfo())


size.percent <- seq(from = .1, to = 1, by = .1)
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
  size0 = round(nrow(data.training[data.training$left == 0,]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$left == 1,]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("left"), size = c(size1,size0) , method = "srswor")
   
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$left))
  
  ##GET CV ERROR 
  ctrl <- trainControl(method="cv",number = 10) 
  
  (model <- train(left ~ ., data.strata, method = "gbm", verbose = FALSE, 
                  trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
  time.cv <- c(time.cv, (model$time$everything["user.self"]))
  
  kappa.cv <- c(kappa.cv, model$results[["Kappa"]])
  accuracy.cv <- c(accuracy.cv, model$results[["Accuracy"]])
  

  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  (model.train <- train(left ~ ., data.strata, method = "gbm", verbose = FALSE, 
                        trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa"))
  time.train <- c(time.train, (model.train$time$everything["user.self"]))
  
  pred.train <- predict(model.train, data.strata, type = "raw")
  cf.train <- confusionMatrix(pred.train, data.strata$left)
  kappa.train <- c(kappa.train, cf.train$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.train$overall[["Accuracy"]])
  
}


##PLOT LEARNING CURVES
plot(size.num, kappa.cv, type = "b", col = "black",  ylim=c(.20,1), xlab = "Data Size", ylab = "Kappa", main = "Learning Curve" )
points(size.num, kappa.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)


##PLOT TIME CURVES
plot(size.num,  time.train, type = "b", col = "red", ylim = c(0,20),##ylim = c(0,0.25), 
     ylab = "Seconds", xlab = "Data Size", main = "Time vs Data Size")
points(size.num, time.cv, type = "b", col = "black" )
legend("topleft", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)





##############################################
###############  NEURAL NET  #################
##############################################

##SET SEED
set.seed(17002)


##ENCODE FACTORS
features <- dummyVars(" ~ department", data = data.train)
features.encoded <- data.frame(predict(features, newdata = data.train))
data.train.encoded <- cbind(data.train[,!(colnames(data.train) %in% c("department"))],features.encoded)
str(data.train.encoded)

features <- dummyVars(" ~ department", data = data.test)
features.encoded <- data.frame(predict(features, newdata = data.test))
data.test.encoded <- cbind(data.test[,!(colnames(data.test) %in% c("department"))],features.encoded)
str(data.test.encoded)

##RELABEL DATA
data.training = data.train.encoded 
data.testing = data.test.encoded

##CREATE PERFORMANCE CURVE
set.seed(7644)
ctrl <- trainControl(method="cv",number = 10) 
size <- c(1,3,5,10,15,25)
decay <- 0
tuneGrid = expand.grid(size = size, decay = decay) ##modelLookup("gbm") ##names(getModelInfo())
(model <- train(left ~ ., data.training, method = "nnet", verbose = FALSE, trControl = ctrl, 
                tuneGrid = tuneGrid,  preProcess = c("center","scale"), metric = "Kappa", maxit = 300))
plot(model, xlab = "Hidden Units", ylab = "Kappa", main = "Performance Curve")

##TEST MODEL 
pred <- predict(model, data.testing)
(cf <- confusionMatrix(pred, data.testing$left))





##PLOT LEARNING CURVE
set.seed(3330033)
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
  size0 = round(nrow(data.training[data.training$left == 0,]) * size.train, 0)
  size1 = round(nrow(data.training[data.training$left == 1,]) * size.train, 0)
  print(size.num <- c(size.num, (size0 + size1)))
  strata.rows <- sampling::strata(data = data.training, stratanames = c("left"), size = c(size1,size0) , method = "srswor")
    
  data.strata <- data.training[strata.rows$ID_unit,]
  print(table(data.strata$left))
  
  ##GET CV ERROR  **GET CV FROM MODEL**
  ctrl <- trainControl(method="cv",number = 10) 
  (model <- train(left ~ ., data.strata, method = model.type, verbose = FALSE, 
                  trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa", 
                  maxit = 300, preProcess = c("center","scale")))
  time.cv <- c(time.cv, (model$time$everything["user.self"]))
  
  kappa.cv.model <- c(kappa.cv.model, model$results[["Kappa"]])
  accuracy.cv.model <- c(accuracy.cv.model, model$results[["Accuracy"]])
  
  
  
  pred <- predict(model, data.strata, type = "raw")
  cf <- confusionMatrix(pred, data.strata$left)
  kappa.cv <- c(kappa.cv, cf$overall[["Kappa"]])
  accuracy.cv <- c(accuracy.cv, cf$overall[["Accuracy"]])
  
  
  ##GET TRAINING ERROR
  ctrl <- trainControl(method="none") 
  (model.train <- train(left ~ ., data.strata, method = model.type, verbose = FALSE, 
                        trControl = ctrl, tuneGrid = tuneGrid, metric = "Kappa", 
                        maxit = 300, preProcess = c("center","scale")))
  time.train <- c(time.train, (model.train$time$everything["user.self"]))
  
  pred.train <- predict(model.train, data.strata, type = "raw")
  cf.train <- confusionMatrix(pred.train, data.strata$left)
  kappa.train <- c(kappa.train, cf.train$overall[["Kappa"]])
  accuracy.train <- c(accuracy.train, cf.train$overall[["Accuracy"]])
  
}

##PLOT CURVES
plot(size.num, kappa.cv.model, type = "b", col = "black",  ylim=c(0.5,1), xlab = "Data Size", ylab = "Kappa", main = "Learning Curve" )
points(size.num, kappa.train, type = "b", col = "red")
legend("bottomright", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)



##PLOT TIME CURVES
plot(size.num,  time.train, type = "b", col = "red", ylim = c(0,200),##ylim = c(0,0.25), 
     ylab = "Seconds", xlab = "Data Size", main = "Time vs Data Size")
points(size.num, time.cv, type = "b", col = "black" )
legend("topleft", inset=.01, title="Legend",
       c("Train","Cross Validation"), fill=c("red","black"), horiz=FALSE)












