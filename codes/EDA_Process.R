library(ggplot2)
library(psych)
library(tidyr)
library(dplyr)
library(corrplot)

MinMax <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

Standard <- function(x, na.rm=TRUE) {
  return((x-mean(x))/sd(x))
}

data <- read.csv(file="stock_train_data.csv")

numeric_var <- data[,!colnames(data) %in% c("id","macd_way")]
numeric_var_features <- numeric_var[,c("rsi", "macd", "signal", "cci", "pctB", "close")]
numeric_var_targets <- numeric_var[,c("avg3", "avg5", "up", "up3", "up5")]

numeric_var_features <- apply(numeric_var_features, 2, MinMax) %>% as.data.frame()
numeric_var_targets <- apply(numeric_var_targets, 2, MinMax) %>% as.data.frame()

mean_table <- apply(numeric_var, 2, function(x) round(mean(x), 2))
sd_table <- apply(numeric_var, 2, function(x) round(sd(x), 2))
mean_sd_table <- as.data.frame(rbind(Mean=mean_table, SD=sd_table))

long_features <- numeric_var_features %>% 
  pivot_longer(
    cols=colnames(numeric_var_features),
    names_to = "var",
    values_to = "value"
  )

long_features$var <- factor(long_features$var, 
                               levels = c("rsi", "macd", "signal", "cci", "pctB", "close"))

long_targets <- numeric_var_targets %>% 
  pivot_longer(
    cols=colnames(numeric_var_targets),
    names_to = "var",
    values_to = "value"
  )

long_targets$var <- factor(long_targets$var, 
                            levels = c("avg3", "avg5", "up", "up3", "up5"))

features_distriburion_plot <- ggplot(long_features, aes(x=var, y=value)) + 
  geom_boxplot() + 
  coord_flip()+
  labs(y="0~1 min-max range", x="Features")+
  theme(
    axis.title = element_text(size=24),
    axis.text = element_text(size=19)
  )

targets_distriburion_plot <- ggplot(long_targets, aes(x=var, y=value)) + 
  geom_boxplot() + 
  coord_flip()+
  labs(y="0~1 min-max range", x="Targets")+
  theme(
    axis.title = element_text(size=24),
    axis.text = element_text(size=19)
  )


pca <- prcomp(~., numeric_var_features, scale=T, center=T)

pca_rotation <- pca$rotation %>% round(3)

cum_var <- cbind(PC=as.factor(1:6), 
                 cummulative_variance=cumsum(pca$sdev^2/sum(pca$sdev^2)))

pca_cumvar_plot <- ggplot(cum_var, aes(x=PC, y=cummulative_variance, group=1))+
  geom_point(size=6)+
  geom_line(size=1)+
  labs(y="cummulative variance", x="Principle Components")+
  theme(
    axis.title = element_text(size=24),
    axis.text = element_text(size=19)
  )+
  scale_x_continuous(breaks=seq(1,6,1))

# write.csv(mean_sd_table, "mean_sd_table.csv")
# write.csv(pca_rotation, "pca_rotation.csv")

save(features_distriburion_plot,
     targets_distriburion_plot,
     pca_cumvar_plot,
     mean_sd_table,
     pca_rotation, file="EDA_result.RData")
