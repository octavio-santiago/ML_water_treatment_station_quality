library(h2o)
library(plyr)
h2o.init()
library(dplyr)

mydata <- read.csv("dfPreTratado.csv", header=TRUE, sep=",")


df <- as.h2o(mydata)


# split into train and validation sets
mydata.split <- h2o.splitFrame(data = df,ratios = 0.80, seed = 1234)
train <- mydata.split[[1]]
valid <- mydata.split[[2]]



predictors <- c('Eq.D.1','phEntrada','phNeutralizacao','phAeracao','OD',
                'Efluente.D.1','VLodoSD30','VAeracao','VEqualizacao')

response <- "Efluente.Final"

#modelo GBM
df_gbm <- h2o.gbm(y = response , x = predictors, training_frame = train,
        ntrees = 50, max_depth = 30, min_rows = 2)

#h2o.confusionMatrix(df_gbm)
h2o.performance(df_gbm, newdata = valid, train = FALSE, valid = TRUE,
                xval = FALSE)
h2o.scoreHistory(df_gbm)
h2o.varimp(df_gbm)

pred = predict(df_gbm,valid)
finalPrediction = as.data.frame(h2o.cbind(valid,pred))
finalPrediction$Diff = finalPrediction$Efluente.Final - finalPrediction$predict


#GLM
df_glmPred <-  h2o.glm(y = response, x = predictors, training_frame = train, family = "gaussian",
                       nfolds = 0, alpha = 0.1, lambda_search = FALSE)


h2o.performance(df_glmPred, newdata = valid, train = FALSE, valid = TRUE,
                xval = FALSE)
h2o.scoreHistory(df_glmPred)
h2o.varimp(df_glmPred)

pred = predict(df_glmPred,valid)
finalPrediction = as.data.frame(h2o.cbind(valid,pred))
finalPrediction$Diff = finalPrediction$Efluente.Final - finalPrediction$predict

