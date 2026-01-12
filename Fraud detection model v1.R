# Core libraries
library(tidyverse)
library(caret)
library(pROC)
library(PRROC)
library(randomForest)
library(xgboost)
library(solitude)
library(dbscan)

set.seed(42)

# ---------- Utility Functions ---------- #

scale_features <- function(df, cols) {
  df[cols] <- scale(df[cols])
  df
}

split_data <- function(df, target, p = 0.8) {
  idx <- createDataPartition(df[[target]], p = p, list = FALSE)
  list(
    train = df[idx, ],
    test  = df[-idx, ]
  )
}

evaluate_model <- function(y_true, probs, thresholds = c(0.5,0.3,0.2,0.1)) {
  roc_obj <- roc(y_true, probs)
  cat("ROC AUC:", auc(roc_obj), "\n")
  
  for (t in thresholds) {
    cat("\nThreshold:", t, "\n")
    preds <- ifelse(probs > t, 1, 0)
    print(confusionMatrix(
      factor(preds),
      factor(y_true),
      positive = "1"
    ))
  }
}

### Data Preparation ###
df <- read.csv("E:/Projects/Credit card Fraud detection dataset.csv", header=TRUE)
df <- scale_features(df, c("Time", "Amount"))

splits <- split_data(df, "Class", 0.8)
train <- splits$train
test  <- splits$test

#### Logistic regression (Baseline + Weighted)
# Baseline Logistic
log_model <- glm(Class ~ ., data = train, family = binomial)
test$log_prob <- predict(log_model, test, type = "response")

evaluate_model(test$Class, test$log_prob)

# Weighted Logistic
fraud_weight <- sum(train$Class == 0) / sum(train$Class == 1)

log_w <- glm(
  Class ~ .,
  data = train,
  family = binomial,
  weights = ifelse(train$Class == 1, fraud_weight, 1)
)

test$log_w_prob <- predict(log_w, test, type = "response")
evaluate_model(test$Class, test$log_w_prob)

#### Random Forest
train$Class <- factor(train$Class)
test$Class  <- factor(test$Class)

rf_model <- randomForest(
  Class ~ .,
  data = train,
  ntree = 300,
  importance = TRUE
)

test$rf_prob <- predict(rf_model, test, type = "prob")[, "1"]
evaluate_model(as.numeric(as.character(test$Class)), test$rf_prob)

varImpPlot(rf_model)


#### xGBoost Model
X <- model.matrix(Class ~ . -1, data = df)
y <- df$Class

train_x <- X[rownames(train), ]
test_x  <- X[rownames(test), ]

dtrain <- xgb.DMatrix(train_x, label = y[rownames(train)])
dtest  <- xgb.DMatrix(test_x,  label = y[rownames(test)])

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 200,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

test$xgb_prob <- predict(xgb_model, dtest)
evaluate_model(y[rownames(test)], test$xgb_prob)

#install.packages(SHAPforxgboost)
#library(SHAPforxgboost)

# Calculate SHAP values for the XGBoost model
#shap_values <- shap.values(xgb_model = xgb_model, X_train = train_x)
#shap_int <- shap.prep(xgb_model = xgb_model, X_train = train_x)

# Plot the most important features
#shap.plot.summary(shap_int)

### Local Outlier Factor
test$lof_score <- lof(test_x, k = 20)

### Hybrid Fraud Detection
xgb_thresh <- 0.1
lof_thresh <- quantile(test$lof_score, 0.995)

test$hybrid_pred <- ifelse(
  test$xgb_prob > xgb_thresh |
    test$lof_score > lof_thresh, 1, 0
)

confusionMatrix(
  factor(test$hybrid_pred),
  factor(test$Class),
  positive = "1"
)

# Function to find the "Business Optimal" Threshold
find_optimal_threshold <- function(probs, true_class, amounts) {
  thresholds <- seq(0.01, 0.5, by = 0.01)
  costs <- sapply(thresholds, function(t) {
    preds <- ifelse(probs > t, 1, 0)
    # Cost of False Positives ($5 each)
    fp_cost <- sum(preds == 1 & true_class == 0) * 5
    # Cost of False Negatives (Actual stolen amount)
    fn_cost <- sum(preds == 0 & true_class == 1 * amounts)
    return(fp_cost + fn_cost)
  })
  
  return(thresholds[which.min(costs)])
}

opt_t <- find_optimal_threshold(test$xgb_prob, test$Class, test$Amount)
cat("The Business-Optimal Threshold is:", opt_t)
