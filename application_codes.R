#### Codes for the application on the river network See Section 4 for
#### more details

library(rpart)
library(e1071)
library(glmnet)
library(randomForest)
source("functions.R")


##### Import and transform data
Data2 <- graphicalExtremes::danube$data_clustered

Data <- apply(Data2, MARGIN = 2, 
              FUN = function(x) (x - min(x))/(max(x) - min(x)))

##################################

X23 <- Data[, 23]
X24 <- Data[, 24]
X1 <- Data[, 1]
thresh <- quantile(X1, probs = 0.85)

eps <- 0.6
v <- eps * thresh

B <- 50  # number of samples for the cross validation

R.lin = R.lin.eps = R.tree = R.tree.eps = R.svml = 
  R.svml.eps = R.lasso = R.lasso.eps = R.rf = R.rf.eps = 
  vector(mode = "list", length = B)

test.lin = test.tree = test.svml = test.rf = 
  test.lin.eps = test.tree.eps = test.svml.eps = test.rf.eps = 
  rep(NA, B)

theta.all = theta.all.eps = matrix(NA, nrow = B, ncol = 2)

X.all <- cbind(X23, X24)

# linear classifier, optim only on the subset C need to be trained
# once for the initial optimization parameters
reslin0 <- lm(X1 ~ X.all)
reslm0 <- reslin0$coefficients[-1]
g_0 <- reslm0 %*% t(X.all)
reslin <- lm(X1 ~ X23 + X24, data = data.frame(X1 = X1[X1 > v & g_0 > v],
                                               X23 = X23[X1 > v & g_0 > v], 
                                               X24 = X24[X1 > v & g_0 > v]))
reslm <- reslin$coefficients[-1]

par(mfrow = c(1, 2))

u <- thresh
for (b in 1:B) {
  ind.train <- sample.int(length(X1), size = 0.7 * length(X1), replace = F)
  
  X_train <- X.all[ind.train, ]
  X_test <- X.all[-ind.train, ]
  
  H_train <- X1[ind.train]
  H_test <- X1[-ind.train]
  Y_train <- 2 * (H_train > u) - 1
  Y_test <- 2 * (H_test > u) - 1
  
  Y_train.eps <- 2 * (H_train > v) - 1
  Y_test.eps <- 2 * (H_test > v) - 1
  
  ##### Linear classifier with mass ######
  theta <- linear.classifier(X = X_train, thresh = u, H = H_train, initials = reslm0,
                             epsilon = 0)$theta
  theta.all[b, ] <- theta
  glin <- 2 * (as.vector(theta %*% t(X_test)) > u) - 1
  
  R.lin[[b]] <- empirical.risk(Y = Y_test, g = glin, epsilon = 0)
  
  ##### Linear classifier without mass ######
  theta.eps <- linear.classifier(X = X_train, thresh = u, H = H_train,
                                 initials = reslm, epsilon = eps)$theta
  theta.all.eps[b, ] <- theta.eps
  glineps <- 2 * (as.vector(theta.eps %*% t(X_test)) > v) - 1
  
  R.lin.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = glin,
                                   g.eps = glineps, epsilon = eps)
  
  
  ##### Regression trees with mass ######
  treeclass <- rpart(y ~ ., data = data.frame(x = X_train, y = as.factor(Y_train)),
                     method = "class")
  gtree <- as.numeric(as.character(predict(treeclass,
                                           newdata = data.frame(x = X_test, y = Y_test), 
                                           type = "class")))
  R.tree[[b]] <- empirical.risk(Y = Y_test, g = gtree, epsilon = 0)
  ##### Regression trees without mass ######
  treeclass.eps <- rpart(y ~ ., data = data.frame(x = X_train, y = as.factor(Y_train.eps)),
                         method = "class")
  gtreeeps <- as.numeric(as.character(predict(treeclass.eps, 
                                              newdata = data.frame(x = X_test, y = Y_test.eps), 
                                              type = "class")))
  R.tree.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = gtree,
                                    g.eps = gtreeeps, epsilon = eps)
  
  #### SVM linear kernel with mass #####
  X_train1 <- X_train
  Y_train1 <- Y_train
  svmclass <- svm(y ~ ., data = data.frame(x = X_train1, y = Y_train1),
                  type = "C-classification", kernel = "linear")
  gsvm <- as.numeric(as.character(predict(svmclass, newdata = data.frame(x = X_test,
                                                                         y = Y_test))))
  R.svml[[b]] <- empirical.risk(Y = Y_test, g = gsvm, epsilon = 0)
  
  #### SVM linear kernel without mass #####
  X_train1 <- X_train
  Y_train.eps1 <- Y_train.eps  # Y_train.eps#
  svmclass.eps <- svm(y ~ ., data = data.frame(x = X_train1, y = Y_train.eps1),
                      type = "C-classification", kernel = "linear")
  gsvmeps <- as.numeric(as.character(predict(svmclass.eps, newdata = data.frame(x = X_test,
                                                                                y = Y_test.eps))))
  R.svml.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = gsvm,
                                    g.eps = gsvmeps, epsilon = eps)
  
  #### Random forest with mass #####
  forest <- randomForest(x = X_train, y = as.factor(Y_train))
  grf <- as.numeric(as.character(predict(forest, newdata = cbind(X_test,
                                                                 Y_test))))
  R.rf[[b]] <- empirical.risk(Y = Y_test, g = grf, epsilon = 0)
  
  #### Random forest without mass #####
  forest.eps <- randomForest(x = X_train, y = as.factor(Y_train.eps))
  grfeps <- as.numeric(as.character(predict(forest.eps, newdata = cbind(X_test,
                                                                        Y_test.eps))))
  R.rf.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = grf,
                                  g.eps = grfeps, epsilon = eps)
  
  #### Logistic regression with mass #####
  X_train1 <- X_train
  Y_train1 <- Y_train
  fraction <- table(Y_train1)/length(Y_train1)
  weights <- 1 - fraction[as.character(Y_train1)]
  m_train1 <- apply(X_train1, 2, mean)
  sd_train1 <- apply(X_train1, 2, sd)
  cv.lasso <- cv.glmnet(x = scale(as.matrix(X_train1)), y = as.factor(Y_train1),
                        weights = weights, family = "binomial")
  glasso <- as.numeric(predict(cv.lasso, scale(as.matrix(X_test), center = m_train1,
                                               scale = sd_train1), type = "class"))
  R.lasso[[b]] <- empirical.risk(Y = Y_test, g = glasso, epsilon = 0)
  
  #### Logistic regression without mass #####
  X_train1 <- X_train
  Y_train1 <- Y_train.eps
  fraction <- table(Y_train1)/length(Y_train1)
  weights <- 1 - fraction[as.character(Y_train1)]
  m_train1 <- apply(X_train1, 2, mean)
  sd_train1 <- apply(X_train1, 2, sd)
  cv.lasso.eps <- cv.glmnet(x = scale(as.matrix(X_train1)), y = as.factor(Y_train1),
                            weights = weights, family = "binomial")
  glasso.eps <- as.numeric(predict(cv.lasso.eps, scale(as.matrix(X_test),
                                                       center = m_train1, scale = sd_train1), 
                                   type = "class"))
  R.lasso.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps,
                                     g = glasso, g.eps = glasso.eps, epsilon = eps)
  
  test.lin[b] <- test.equal.risk(Y = Y_test, g1 = glin, g2 = glasso,
                                 alternative = "greater", epsilon = 0)
  test.lin.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps,
                                     g1 = glin, g1.eps = glineps, g2 = glasso, g2.eps = glasso.eps,
                                     alternative = "greater", epsilon = eps)
  test.rf[b] <- test.equal.risk(Y = Y_test, g1 = grf, g2 = glasso, alternative = "greater",
                                epsilon = 0)
  test.rf.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps, g1 = grf,
                                    g1.eps = grfeps, g2 = glasso, g2.eps = glasso.eps, 
                                    alternative = "greater",
                                    epsilon = eps)
  test.svml[b] <- test.equal.risk(Y = Y_test, g1 = gsvm, g2 = glasso,
                                  alternative = "greater", epsilon = 0)
  test.svml.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps,
                                      g1 = gsvm, g1.eps = gsvmeps, g2 = glasso, g2.eps = glasso.eps,
                                      alternative = "greater", epsilon = eps)
  
  test.tree[b] <- test.equal.risk(Y = Y_test, g1 = gtree, g2 = glasso,
                                  alternative = "greater", epsilon = 0)
  test.tree.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps,
                                      g1 = gtree, g1.eps = gtreeeps, g2 = glasso, g2.eps = glasso.eps,
                                      alternative = "greater", epsilon = eps)
  
}


## Figure 4 in Section 4
par(mfrow = c(1, 2))
boxplot(cbind(unlist(lapply(R.lin, `[[`, 1)), unlist(lapply(R.tree, `[[`, 1)), 
              unlist(lapply(R.svml, `[[`, 1)), unlist(lapply(R.lasso, `[[`, 1)), 
              unlist(lapply(R.rf, `[[`, 1))), ylab = "Risk", cex.lab = 2, cex.axis = 2,
        cex.main = 2, cex.sub = 2, cex = 1.5, ylim = c(0, 1), xaxt = "n", col = "#DC3220")
axis(1, at = c(1, 2, 3, 4, 5), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "Lasso", "RF"), cex.axis = 1.5)

boxplot(cbind(unlist(lapply(R.lin.eps, `[[`, 1)), unlist(lapply(R.tree.eps, `[[`, 1)), 
              unlist(lapply(R.svml.eps, `[[`, 1)), unlist(lapply(R.lasso.eps, `[[`, 1)), 
              unlist(lapply(R.rf.eps, `[[`, 1))), ylab = "Risk", cex.lab = 2,
        cex.axis = 2, cex.main = 2, cex.sub = 2, cex = 1.5, ylim = c(0, 1),
        xaxt = "n", col = "#005AB5")
axis(1, at = c(1, 2, 3, 4, 5), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "Lasso", "RF"), cex.axis = 1.5)


## Figure 6 in Appendix C
boxplot(cbind(test.lin, test.tree, test.svml, test.rf), ylab = "p-value",
        cex.lab = 1.8, cex.axis = 2, cex.main = 2, cex.sub = 2, cex = 1.5,
        ylim = c(0, 1), xaxt = "n", col = "#DC3220")
axis(1, at = c(1, 2, 3, 4), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "RF"), cex.axis = 1.5)
abline(h = 0.05, lwd = 2, lty = "dotted")

boxplot(cbind(test.lin.eps, test.tree.eps, test.svml.eps, test.rf.eps),
        ylab = "p-value", cex.lab = 1.8, cex.axis = 2, cex.main = 2,
        cex.sub = 2, cex = 1.5, ylim = c(0, 1), xaxt = "n", col = "#005AB5")
axis(1, at = c(1, 2, 3, 4), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "RF"), cex.axis = 1.5)
abline(h = 0.05, lwd = 2, lty = "dotted")