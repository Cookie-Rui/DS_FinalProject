# Import packages
library(randomForest)
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
library(lubridate)
library(quantmod)
library(TTR)
library(zoo)

# The stock Id for crawling the data (in case needed)
ALL_STOCK_CODES <- c("3481.TW","6488.TWO","5608.TW","2329.TW","3050.TW",
                 "2603.TW","2615.TW","3037.TW","2882.TW","2382.TW","3035.TW","2881.TW",
                 "1513.TW","8069.TWO","2317.TW","2330.TW","2609.TW","6182.TWO","3707.TWO",
                 "5425.TWO","6104.TWO","2201.TW","1795.TW","5347.TWO","1301.TW","1216.TW",
                 "1102.TW","9958.TW","2542.TW","4128.TWO","5876.TW","2915.TW","6806.TW",
                 "6505.TW","2731.TW","8255.TWO","1477.TW","5609.TWO","6111.TWO","3687.TWO",
                 "8938.TWO","1903.TW","2732.TWO","5508.TWO","5904.TWO","8349.TWO","6582.TW")

features_WithID <- c("id","rsi","macd","cci","pctB","macd_way","signal","close")
  
# c("id","rsi","macd","cci","bb_pctB","bb_dn","bb_mavg","bb_up",
# "tdi","cmo","cti","kst","vol","obv","mfi","wpr","roc","trix","clv","adx",
# "macd_way","ma3","ma5","ma15","ma30","ma60","ma180")

features_WithoutID <- c("rsi","macd","cci","pctB","macd_way","signal","close")
  
# c("rsi","macd","cci","bb_pctB","bb_dn","bb_mavg","bb_up",
# "tdi","cmo","cti","kst","vol","obv","mfi","wpr","roc",
# "trix","clv","adx","macd_way","ma3","ma5","ma15","ma30",
# "ma60","ma180")

# Crawling for stock data, stock_codes can be either single or multiple
# This function is not needed if using the original data
Make_Stock_Data <- function(stock_codes, result_data){
    # Traversing the stock_codes list
    for (stock_code in stock_codes){
        # Download stock data
        stock <- getSymbols(stock_code, src = "yahoo", 
                            from = today - years(4), to = Sys.Date())
        og_data <- get(stock)[,1:5]
        colnames(og_data) <- c("Open", "High", "Low", "Close", "Volume")
        og_data <- na.omit(og_data)
        
        # Computing indexes
        close <- Cl(og_data)
        rsi <- RSI(og_data[,"Close"])
        macd <- MACD(og_data[,"Close"])
        cci <- CCI(og_data[,c("High","Low","Close")])
        bbands <- BBands(og_data[,c("High","Low","Close")])
        # tdi <- TDI(og_data[,"Close"])
        # cmo <- CMO(og_data[,"Close"])
        # cti <- CTI(og_data[,"Close"])
        # dpo <- DPO(og_data[,"Close"])
        # dvi <- DVI(og_data[,"Close"]) # too much NA, not put in data
        # kst <- KST(og_data[,"Close"])
        # vol <- volatility(og_data[,c("Open","High","Low","Close")])
        # obv <- OBV(og_data[,"Close"], og_data[,"Volume"])
        # mfi <- MFI(og_data[,c("High","Low","Close")], og_data[,"Volume"])
        # wpr<- WPR(og_data[,c("High","Low","Close")])
        # roc <- ROC(og_data[,"Close"])
        # trix  <- TRIX(og_data[,"Close"])
        # clv <- CLV(og_data[,c("High","Low","Close")])
        # adx <- ADX(og_data[,c("High","Low","Close")])
        # MA3 <- SMA(og_data[,"Close"], 3)
        # MA5 <- SMA(og_data[,"Close"], 5)
        # MA15 <- SMA(og_data[,"Close"], 15)
        # MA30 <- SMA(og_data[,"Close"], 30)
        # MA60 <- SMA(og_data[,"Close"], 60)
        # MA180 <- SMA(og_data[,"Close"], 180)
        
        # Create data frame
        df <- data.frame(close = close,
                         rsi = rsi, 
                         macd = macd$macd,
                         cci=cci,
                         bb_pctB=bbands$pctB)
                         # bb_dn=bbands$dn,
                         # bb_mavg=bbands$mavg,
                         # bb_up=bbands$up,
                         # tdi=tdi$tdi,
                         # cmo=cmo,
                         # cti=cti,
                         # kst=kst$kst,
                         # vol=vol,
                         # obv=obv,
                         # mfi=mfi,
                         # wpr=wpr$Close,
                         # roc=roc$Close,
                         # trix=trix$TRIX,
                         # clv=clv,
                         # adx=adx$ADX,
                         # ma3=MA3,
                         # ma5=MA5,
                         # ma15=MA15,
                         # ma30=MA30,
                         # ma60=MA60,
                         # ma180=MA180
        
        # colnames(df)[c(4:7,15,16,17,19,20:26)] <- c("bb_pctB","bb_dn","bb_mavg","bb_up",
        #                                             "wpr","roc","trix","adx","close","ma3",
        #                                             "ma5","ma15","ma30","ma60","ma180")
        
        # Imterpolation
        df <- na.approx(df)
        df <- as.data.frame(df)
        
        
        # Add stock Id
        df$id <- stock_code
        
        # Add new columns for price movements
        df[, c("up", "avg5", "up5","avg3","up3")] <- NA

        # Calculate price-related features
        for (i in 1:(nrow(df) - 5)) {
            df[i, "up"] <- df[i + 1, "Close"] - df[i, "Close"]
            df[i, "avg5"] <- mean(df[(i + 1):(i + 5), "Close"])
            df[i, "up5"] <- df[i, "avg5"] - df[i, "Close"]
            df[i,"avg3"]<- mean(df[(i + 1):(i + 3), "Close"])
            df[i,"up3"]<-df[i, "avg3"] - df[i, "Close"]
        }
        
        
        # Add a new column for MACD direction
        df$macd_way <- ifelse(df$macd > macd$signal, "up", "down")
        # df$date <- rownames(df)
        
        result_data <- rbind(result_data, df)
    }
    
    result_data <- na.omit(result_data)
    return(result_data)   
}


# Doing some preprocess and train_test_split
Data_Preprocess <- function(data, target, feature_select=NULL,
                            Kfolds=5, macd_to_numeric=TRUE){
  
  # Transform the "up" variables to binary features
  data$up <- ifelse(data$up>=0, 1, 0)
  data$up3 <- ifelse(data$up3>=0, 1, 0)
  data$up5 <- ifelse(data$up5>=0, 1, 0)
  
  # Transform "macd_way" to numeric binary features
  if(macd_to_numeric == TRUE){
    data$macd_way[data$macd_way=='up'] <- 1
    data$macd_way[data$macd_way=='down'] <- 0
    data$macd_way <- as.numeric(data$macd_way)
  }
  
  # Select the features and the target we want
  if(!is.null(feature_select)){
    data <- data[c(feature_select, target)]
  }
  
  # Split the data by stock Id
  folds <- groupKFold(data$id, k = Kfolds)
  train_indices <- folds[[1]]
  dtrain <- data[train_indices, ]
  dtest <- data[-train_indices, ]
  
  # Remove stock Id feature 
  dtrain$id <- NULL
  dtest$id <- NULL
  
  # Move our target variable to the first column of the data
  dtrain <- dtrain %>% dplyr::relocate(target)
  dtest <- dtest %>% dplyr::relocate(target)
  
  return(list(train=dtrain, test=dtest))
}


# Doing the same thing as "Data_Preprocess()", but with PCA
Data_Preprocess_WIth_PCA <- function(data, target, feature_select=NULL,
                            Kfolds=5, macd_to_numeric=TRUE, num_pca){
  data$up <- ifelse(data$up>=0, 1, 0)
  data$up3 <- ifelse(data$up3>=0, 1, 0)
  data$up5 <- ifelse(data$up5>=0, 1, 0)
  
  if(macd_to_numeric == TRUE){
    data$macd_way[data$macd_way=='up'] <- 1
    data$macd_way[data$macd_way=='down'] <- 0
    data$macd_way <- as.numeric(data$macd_way)
  }
  
  if(!is.null(feature_select)){
    data <- data[c(feature_select, target)]
  }
  
  id <- data$id
  data$id <- NULL
  pca <- prcomp(~., data=data[!names(data) %in% c(target)], scale=T, center=T)
  data <- cbind(pca$x[,1:num_pca], data[target])
  data <- cbind(data, id)
  
  folds <- groupKFold(data$id, k = Kfolds)
  train_indices <- folds[[1]]
  dtrain <- data[train_indices, ]
  dtest <- data[-train_indices, ]
  
  dtrain$id <- NULL
  dtest$id <- NULL
  
  dtrain <- dtrain %>% dplyr::relocate(target)
  dtest <- dtest %>% dplyr::relocate(target)
  
  return(list(train=dtrain, test=dtest))
}


# Transform the train/test data into XGB.MAtrix objects
To_xgbDMatrix <- function(train, test){
  xgb_train <- xgb.DMatrix(data = as.matrix(train[,-1])
                          , label = train[[1]])
  
  xgb_test <- xgb.DMatrix(data = as.matrix(test[,-1])
                            , label = test[[1]])
  
  watchlist <- list(train=xgb_train, test=xgb_test)
  
  return(list(xgb_train=xgb_train, xgb_test=xgb_test, watchlist=watchlist))
}


# Calculate metrics for regression model
Reg_Metrics <- function(obs, pred){
  r2 <- R2(obs=obs, pred=pred)
  mse <- mse(obs, pred)
  rmse <- rmse(obs, pred)
  mae <- mae(obs, pred)
  summary <- data.frame(R2=r2, MSE=mse, RMSE=rmse, MAE=mae)
  table <- data.frame(Actual=obs, Predict=pred)
  return(list(summary=summary, table=table))
}


# Summarize the change of RMSE during the training of regression model
# Return the visualized plot of the change of RMSE
RMSE_Info <- function(model, plot_title){
  rmse_table <- model$evaluation_log %>% 
    dplyr::select(-c("train_logloss", "test_logloss")) %>% 
    pivot_longer(cols = c("train_rmse", "test_rmse"), 
                 names_to = "split", values_to = "RMSE")
  
  plot <- ggplot(data=rmse_table)+
    geom_point(aes(x=iter, y=RMSE, color=split, fill=split))+
    geom_line(aes(x=iter, y=RMSE, color=split))+
    coord_cartesian(ylim = c(0, 25)) +theme_minimal()+
    labs(title = plot_title)
  
  return(list(rmse_table=rmse_table, plot=plot))
}


# Change the "probability" into the "class"
Class_Pred_Transform <- function(pred.xg, up_factor_levels){
  pred.xg <- as.data.frame(pred.xg)
  colnames(pred.xg) = up_factor_levels
  return(apply(pred.xg,1,function(x) as.numeric(colnames(pred.xg)[which.max(x)])))
}


# Calculate metrics for classification model
Class_Metrics <- function(obs, pred){
  table <- data.frame(Actual=obs, Predict=pred)
  confu <- confusionMatrix(factor(obs), factor(pred), 
                             mode = "everything", positive="1")
  acc <- confu$overall["Accuracy"]
  acc_pvalue <- confu$overall["AccuracyPValue"]
  f1 <- confu$byClass['F1']
  summary <- data.frame(Accuracy=acc, ACC_Pvalue=acc_pvalue, F1=f1)
  rownames(summary) <- 1
  return(list(summary=summary, table=table))
}


# Summarize the change of AUC during the training of classification model
# Return the visualized plot of the change of AUC
AUC_Info <- function(model, plot_title){
  auc_table <- model$evaluation_log %>% 
    dplyr::select(-c("train_merror", "test_merror")) %>% 
    pivot_longer(cols = c("train_auc", "test_auc"), 
                 names_to = "split", values_to = "AUC")
  
  plot <- ggplot(data=auc_table)+
    geom_point(aes(x=iter, y=AUC, color=split, fill=split))+
    geom_line(aes(x=iter, y=AUC, color=split))+
    coord_cartesian(ylim = c(0.3, 1)) +theme_minimal()+
    labs(title = plot_title)
  
  return(list(auc_table=auc_table, plot=plot))
}
