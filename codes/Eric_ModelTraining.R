library(zeallot)
library(dplyr)
library(magrittr)
library(ggplot2)
library(caret)
library(ROCR)
library(xgboost)
library(data.table)
library(Metrics)
library(tidyr)
library(patchwork)

source("functions.R")

data <- read.csv('./stock_train_data.csv',header = TRUE)

# null_df <- data.frame()
# data <- Make_Stock_Data("6488.TWO", null_df)

set.seed(530)

c(avg3_train, avg3_test) %<-% Data_Preprocess(data, "avg3", 
                                            feature_select=features_WithID, 
                                            Kfolds=5, macd_to_numeric=TRUE)

c(avg5_train, avg5_test) %<-% Data_Preprocess(data, "avg5", 
                                            feature_select=features_WithID, 
                                            Kfolds=5, macd_to_numeric=TRUE)

c(up3_train, up3_test) %<-% Data_Preprocess(data, "up3", 
                                            feature_select=features_WithID, 
                                            Kfolds=5, macd_to_numeric=TRUE)

c(up5_train, up5_test) %<-% Data_Preprocess(data, "up5", 
                                            feature_select=features_WithID, 
                                            Kfolds=5, macd_to_numeric=TRUE)

c(up_train, up_test) %<-% Data_Preprocess(data, "up", 
                                            feature_select=features_WithID, 
                                            Kfolds=5, macd_to_numeric=TRUE)

# c(avg3_pca_train, avg3_pca_test) %<-% 
#   Data_Preprocess_WIth_PCA(data, "avg3", 
#                            feature_select=features_WithID,
#                            Kfolds=5, macd_to_numeric=TRUE, num_pca=3)
# 
# c(avg5_pca_train, avg5_pca_test) %<-% 
#   Data_Preprocess_WIth_PCA(data, "avg5", 
#                            feature_select=features_WithID, 
#                            Kfolds=5, macd_to_numeric=TRUE, num_pca=3)
# 
# c(up3_pca_train, up3_pca_test) %<-% 
#   Data_Preprocess_WIth_PCA(data, "up3", 
#                            feature_select=features_WithID, 
#                            Kfolds=5, macd_to_numeric=TRUE, num_pca=3)
# 
# c(up5_pca_train, up5_pca_test) %<-% 
#   Data_Preprocess_WIth_PCA(data, "up5", 
#                            feature_select=features_WithID, 
#                            Kfolds=5, macd_to_numeric=TRUE, num_pca=3)
# 
# c(up_pca_train, up_pca_test) %<-% 
#   Data_Preprocess_WIth_PCA(data, "up", 
#                            feature_select=features_WithID, 
#                            Kfolds=5, macd_to_numeric=TRUE, num_pca=3)

up_factor_levels <- levels(factor(up_train$up))


# ------------------------------------------------------------------------
# xgboost

c(xgb_avg3, xgb_avg3_t, avg3_watch) %<-% To_xgbDMatrix(avg3_train, avg3_test)
c(xgb_avg5, xgb_avg5_t, avg5_watch) %<-% To_xgbDMatrix(avg5_train, avg5_test)
c(xgb_up3, xgb_up3_t, up3_watch) %<-% To_xgbDMatrix(up3_train, up3_test)
c(xgb_up5, xgb_up5_t, up5_watch) %<-% To_xgbDMatrix(up5_train, up5_test)
c(xgb_up, xgb_up_t, up_watch) %<-% To_xgbDMatrix(up_train, up_test)

boost_avg3 <- xgb.train(data=xgb_avg3, nround=600, 
                   watchlist=avg3_watch, eval.metric = "rmse",
                   eval.metric = "logloss",
                   objective = "reg:squarederror")

boost_avg5 <- xgb.train(data=xgb_avg5, nround=600, 
                   watchlist=avg5_watch, eval.metric = "rmse",
                   eval.metric = "logloss",
                   objective = "reg:squarederror")

boost_up3 <- xgb.train(data=xgb_up3, nround=600, 
                   watchlist=up3_watch, eval.metric = "auc",
                   eval.metric = "merror",
                   objective = "multi:softprob",
                   num_class=length(up_factor_levels))

boost_up5 <- xgb.train(data=xgb_up5, nround=600, 
                   watchlist=up5_watch, eval.metric = "auc",
                   eval.metric = "merror",
                   objective = "multi:softprob",
                   num_class=length(up_factor_levels))

boost_up <- xgb.train(data=xgb_up, nround=600, 
                   watchlist=up_watch, eval.metric = "auc",
                   eval.metric = "merror",
                   objective = "multi:softprob",
                   num_class=length(up_factor_levels))


pred.xg.avg3 <- predict(boost_avg3, xgb_avg3_t,reshape=T)
pred.xg.avg5 <- predict(boost_avg5, xgb_avg5_t,reshape=T)
pred.xg.up3 <- predict(boost_up3, xgb_up3_t,reshape=T)
pred.xg.up3 <- Class_Pred_Transform(pred.xg.up3, up_factor_levels)
pred.xg.up5 <- predict(boost_up5, xgb_up5_t,reshape=T)
pred.xg.up5 <- Class_Pred_Transform(pred.xg.up5, up_factor_levels)
pred.xg.up <- predict(boost_up, xgb_up_t,reshape=T)
pred.xg.up <- Class_Pred_Transform(pred.xg.up, up_factor_levels)

c(avg3_summary, avg3_result) %<-% Reg_Metrics(avg3_test$avg3, pred.xg.avg3)
c(avg5_summary, avg5_result) %<-% Reg_Metrics(avg5_test$avg5, pred.xg.avg5)
c(up3_summary, up3_result) %<-% Class_Metrics(up3_test$up3, pred.xg.up3)
c(up5_summary, up5_result) %<-% Class_Metrics(up5_test$up5, pred.xg.up5)
c(up_summary, up_result) %<-% Class_Metrics(up_test$up, pred.xg.up)

c(avg3_rmse_table, avg3_rmse_plot) %<-% RMSE_Info(boost_avg3, "avg3")
c(avg5_rmse_table, avg5_rmse_plot) %<-% RMSE_Info(boost_avg5, "avg5")
c(up3_auc_table, up3_auc_plot) %<-% AUC_Info(boost_up3, "up3")
c(up5_auc_table, up5_auc_plot) %<-% AUC_Info(boost_up5, "up5")
c(up_auc_table, up_auc_plot) %<-% AUC_Info(boost_up, "up")

# (avg3_rmse_plot+avg5_rmse_plot)/(up3_auc_plot+up5_auc_plot)/(up_auc_plot)

save(boost_avg3, boost_avg5, boost_up3, boost_up5, boost_up,
     file="Eric_Five_Models.RData")

save(avg3_summary, avg5_summary, up3_summary, up5_summary, up_summary,
     file="Model_MetricsSummary.RData")

save(avg3_result, avg5_result, up3_result, up5_result, up_result,
     file="Predict_Results.RData")

save(avg3_rmse_plot, avg5_rmse_plot, up3_auc_plot, up5_auc_plot, up_auc_plot,
     file="Metrics_Plots.RData")

write.csv(avg3_summary, "avg3_summary.csv")
write.csv(avg3_summary, "avg5_summary.csv")
write.csv(avg3_summary, "up3_summary.csv")
write.csv(avg3_summary, "up5_summary.csv")
write.csv(avg3_summary, "up_summary.csv")