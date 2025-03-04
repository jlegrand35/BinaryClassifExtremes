#### Codes for the simulation part
#### See Section 3 for more details

library(rpart)
library(e1071)
library(glmnet)
library(randomForest)
source("functions.R")

##################################
# Data simulation, see Section 3.1 for details
set.seed(3333)
nsim <- 10000  # simulation sample size
alpha1 <- 3
alpha2 <- 2
X1 <- 1/(runif(nsim)^(1/alpha1))
N <- 1/(runif(nsim)^(1/alpha2))
H <- X1 + N
X2 <- 1/(runif(nsim)^(1/alpha2))
X3 <- rexp(nsim, 1)
X4 <- rexp(nsim, 2)

# Define thresholds u and v=0.4*u
u <- quantile(H, probs = 0.97)
eps <- 0.4
v <- u * eps

X.all <- cbind(X1, X2, X3, X4)

# linear classifier, optim only on the subset C (see Append. B for details)
reslin0 <- lm(H ~ X.all[, c(1, 2)])
reslm0 <- reslin0$coefficients[-1]
g_0 <- reslm0 %*% t(X.all[, c(1, 2)])
reslin <- lm(H ~ X1 + X2, 
             data = data.frame(H = H[H > v & g_0 > v], 
                               X1 = X1[H > v & g_0 > v], 
                               X2 = X2[H > v & g_0 > v]))
reslm <- reslin$coefficients[-1]

B <- 50  # number of samples for the cross validation

R.lin = R.lin.eps = R.tree = R.tree.eps = R.svml = R.svml.eps = 
  R.lasso = R.lasso.eps = R.rf = R.rf.eps = vector(mode = "list", length = B)

Emp.lin = Emp.lin.eps = Emp.lin.eps.2 = Emp.tree = Emp.tree.eps = Emp.svml = 
  Emp.svml.eps = Emp.lasso = Emp.lasso.eps = Emp.rf = Emp.rf.eps = 
  vector(mode = "list", length = B)

theta.all = theta.all.eps = theta.all.eps.2 = matrix(NA, nrow = B, ncol = dim(X.all)[2])

test.lasso = test.tree = test.svml = test.rf = test.lasso.eps = test.tree.eps = 
  test.svml.eps = test.rf.eps = rep(NA, B)

for (b in 1:B) {
  ##### Split data training-validation sets #####
  ind.train <- sample.int(length(H), size = 0.7 * length(H), replace = F)
  
  X_train <- X.all[ind.train, ]
  X_test <- X.all[-ind.train, ]
  
  H_train <- H[ind.train]
  H_test <- H[-ind.train]
  Y_train <- 2 * (H_train > u) - 1
  Y_test <- 2 * (H_test > u) - 1
  
  Y_train.eps <- 2 * (H_train > v) - 1
  Y_test.eps <- 2 * (H_test > v) - 1
  
  ##### Linear classifier with mass ######
  theta <- linear.classifier(X = X_train[, c(1, 2)], thresh = u, H = H_train,
                             initials = reslm0, epsilon = 0)$theta
  theta.all[b, ] <- theta
  glin <- 2 * (as.vector(theta %*% t(X_test[, c(1, 2)])) > u) - 1
  
  R.lin[[b]] <- empirical.risk(Y = Y_test, g = glin, epsilon = 0)
  Emp.lin[[b]] <- sum(Y_test != glin)/length(Y_test)
  
  ##### Linear classifier without mass ######
  theta.eps <- linear.classifier(X = X_train[, c(1, 2)], thresh = u,
                                 H = H_train, initials = reslm, epsilon = eps)$theta
  theta.all.eps[b, ] <- theta.eps
  glineps <- 2 * (as.vector(theta.eps %*% t(X_test[, c(1, 2)])) > v) - 1
  
  R.lin.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = glin,
                                   g.eps = glineps, epsilon = eps)
  
  ##### Regression trees with mass ######
  treeclass <- rpart(y ~ ., data = data.frame(x = X_train, y = as.factor(Y_train)),
                     method = "class")
  gtree <- as.numeric(as.character(predict(treeclass, 
                                           newdata = data.frame(x = X_test, y = Y_test), 
                                           type = "class")))
  R.tree[[b]] <- empirical.risk(Y = Y_test, g = gtree, epsilon = 0)
  Emp.tree[[b]] <- sum(Y_test != gtree)/length(Y_test)
  
  ##### Regression trees without mass ######
  treeclass.eps <- rpart(y ~ ., data = data.frame(x = X_train, y = as.factor(Y_train.eps)),
                         method = "class")
  gtreeeps <- as.numeric(as.character(predict(treeclass.eps, 
                                              newdata = data.frame(x = X_test, y = Y_test.eps), 
                                              type = "class")))
  R.tree.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = gtree,
                                    g.eps = gtreeeps, epsilon = eps)
  
  #### SVM linear kernel with mass #####
  w <- which(Y_train == 1)
  X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
  Y_train1 <- c(Y_train[w], Y_train[1:length(w)])
  svmclass <- svm(y ~ ., data = data.frame(x = X_train1, y = Y_train1), 
                  type = "C-classification", kernel = "linear")
  gsvm <- as.numeric(as.character(predict(svmclass, 
                                          newdata = data.frame(x = X_test, y = Y_test))))
  R.svml[[b]] <- empirical.risk(Y = Y_test, g = gsvm, epsilon = 0)
  Emp.svml[[b]] <- sum(Y_test != gsvm)/length(Y_test)
  
  #### SVM linear kernel without mass #####
  svmclass.eps <- svm(y ~ ., data = data.frame(x = X_train, y = Y_train.eps),
                      type = "C-classification", kernel = "linear")
  gsvmeps <- as.numeric(as.character(predict(svmclass.eps, 
                                             newdata = data.frame(x = X_test, y = Y_test.eps))))
  R.svml.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = gsvm,
                                    g.eps = gsvmeps, epsilon = eps)
  
  #### Random forest with mass #####
  forest <- randomForest(x = X_train, y = as.factor(Y_train))
  grf <- as.numeric(as.character(predict(forest, newdata = cbind(X_test, Y_test))))
  R.rf[[b]] <- empirical.risk(Y = Y_test, g = grf, epsilon = 0)
  Emp.rf[[b]] <- sum(Y_test != grf)/length(Y_test)
  
  #### Random forest without mass #####
  forest.eps <- randomForest(x = X_train, y = as.factor(Y_train.eps))
  grfeps <- as.numeric(as.character(predict(forest.eps, 
                                            newdata = cbind(X_test, Y_test.eps))))
  R.rf.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = grf,
                                  g.eps = grfeps, epsilon = eps)
  
  #### Logistic regression with mass #####
  w <- which(Y_train == 1)
  X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
  Y_train1 <- c(Y_train[w], Y_train[1:length(w)])
  fraction <- table(Y_train1)/length(Y_train1)
  weights <- 1 - fraction[as.character(Y_train1)]
  m_train1 <- apply(X_train1, 2, mean)
  sd_train1 <- apply(X_train1, 2, sd)
  cv.lasso <- cv.glmnet(x = scale(as.matrix(X_train1)), y = as.factor(Y_train1),
                        weights = weights, family = "binomial")
  glasso <- as.numeric(predict(cv.lasso, scale(as.matrix(X_test), center = m_train1,
                                               scale = sd_train1), type = "class"))
  R.lasso[[b]] <- empirical.risk(Y = Y_test, g = glasso, epsilon = 0)
  Emp.lasso[[b]] <- sum(Y_test != glasso)/length(Y_test)
  
  #### Logistic regression without mass #####
  w <- which(Y_train.eps == 1)
  X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
  Y_train1 <- c(Y_train.eps[w], Y_train.eps[1:length(w)])
  fraction <- table(Y_train1)/length(Y_train1)
  weights <- 1 - fraction[as.character(Y_train1)]
  m_train1 <- apply(X_train1, 2, mean)
  sd_train1 <- apply(X_train1, 2, sd)
  cv.lasso.eps <- cv.glmnet(x = scale(as.matrix(X_train1)), y = as.factor(Y_train1),
                            weights = weights, family = "binomial")
  glassoeps <- as.numeric(predict(cv.lasso.eps, 
                                  scale(as.matrix(X_test), center = m_train1, scale = sd_train1), 
                                  type = "class"))
  R.lasso.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps, g = glasso,
                                     g.eps = glassoeps, epsilon = eps)
  
  #### Statistical tests between classifiers (see Appendix C) #### 
  
  test.lasso[b] <- test.equal.risk(Y = Y_test, g1 = glasso, g2 = glin, 
                                   alternative = "greater", epsilon = 0)
  test.lasso.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps, g1 = glasso,
                                       g1.eps = glassoeps, g2 = glin, g2.eps = glineps, 
                                       alternative = "greater", epsilon = eps)
  
  test.rf[b] <- test.equal.risk(Y = Y_test, g1 = grf, g2 = glin, 
                                alternative = "greater", epsilon = 0)
  test.rf.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps, g1 = grf,
                                    g1.eps = grfeps, g2 = glin, g2.eps = glineps, 
                                    alternative = "greater", epsilon = eps)
  
  test.svml[b] <- test.equal.risk(Y = Y_test, g1 = gsvm, g2 = glin, 
                                  alternative = "greater", epsilon = 0)
  test.svml.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps, g1 = gsvm,
                                      g1.eps = gsvmeps, g2 = glin, g2.eps = glineps,
                                      alternative = "greater",  epsilon = eps)
  
  test.tree[b] <- test.equal.risk(Y = Y_test, g1 = gtree, g2 = glin, 
                                  alternative = "greater", epsilon = 0)
  test.tree.eps[b] <- test.equal.risk(Y = Y_test, Y.eps = Y_test.eps, g1 = gtree,
                                      g1.eps = gtreeeps, g2 = glin, g2.eps = glineps, 
                                      alternative = "greater", epsilon = eps)
}

## Figure 5 in Appendix C
par(mfrow=c(1,2))
boxplot(cbind(test.lasso, test.tree, test.svml, test.rf), ylab = "p-value",
        cex.lab = 1.8, cex.axis = 2, cex.main = 2, cex.sub = 2, cex = 1.5,
        xaxt = "n", col = "#DC3220")
axis(1, at = c(1, 2, 3, 4), cex = 1.5, lwd = 2, 
     labels = c("Lasso", "Tree", "SVM", "RF"), cex.axis = 1.5)
abline(h = 0.05, lwd = 2, lty = "dotted")

boxplot(cbind(test.lasso.eps, test.tree.eps, test.svml.eps, test.rf.eps),
        ylab = "p-value", cex.lab = 1.8, cex.axis = 2, cex.main = 2, cex.sub = 2,
        cex = 1.5, ylim = c(0, 1), xaxt = "n", col = "#005AB5")
axis(1, at = c(1, 2, 3, 4), cex = 1.5, lwd = 2, 
     labels = c("Lasso", "Tree", "SVM", "RF"), cex.axis = 1.5)
abline(h = 0.05, lwd = 2, lty = "dotted")

## First row of Figure 3, Section 3
par(mfrow = c(2, 2), mar = c(4.3, 5.1, 2.3, 2.1))
boxplot(cbind(unlist(lapply(R.lin, `[[`, 1)), 
              unlist(lapply(R.tree, `[[`, 1)), 
              unlist(lapply(R.svml, `[[`, 1)), 
              unlist(lapply(R.lasso, `[[`, 1)), 
              unlist(lapply(R.rf, `[[`, 1))), 
        ylab = "Risk", cex.lab = 2, cex.axis = 2,
        cex.main = 2, cex.sub = 2, cex = 1.5, ylim = c(0, 1), 
        xaxt = "n", main = bquote(~v[0] == 0), col = "#DC3220")
axis(1, at = c(1, 2, 3, 4, 5), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "Lasso", "RF"), cex.axis = 2)

boxplot(cbind(unlist(lapply(R.lin.eps, `[[`, 1)), 
              unlist(lapply(R.tree.eps, `[[`, 1)), 
              unlist(lapply(R.svml.eps, `[[`, 1)), 
              unlist(lapply(R.lasso.eps, `[[`, 1)), 
              unlist(lapply(R.rf.eps, `[[`, 1))), 
        ylab = "Risk", cex.lab = 2, cex.axis = 2, cex.main = 2, cex.sub = 2, 
        cex = 1.5, ylim = c(0, 1), xaxt = "n", col = "#005AB5", 
        main = bquote(~epsilon[1] == .(eps) ~ u ~ "&" ~ n[v[1]] == .(sum(H_test > v))))
axis(1, at = c(1, 2, 3, 4, 5), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "Lasso", "RF"), cex.axis = 2)

## Similar but for two other values of v
for (eps_val in c(0.6, 0.8))
{
  k <- 2
  eps <- eps_val
  v <- u * eps
  # linear classifier, optim only on the subset C
  reslin0 <- lm(H ~ X.all[, c(1, 2)])
  reslm0 <- reslin0$coefficients[-1]
  g_0 <- reslm0 %*% t(X.all[, c(1, 2)])
  reslin <- lm(H ~ X1 + X2, 
               data = data.frame(H = H[H > v & g_0 > v], X1 = X1[H > v & g_0 > v],
                                 X2 = X2[H > v & g_0 > v]))
  reslm <- reslin$coefficients[-1]
  R.lin = R.lin.eps = R.tree = R.tree.eps = R.svml = R.svml.eps = 
    R.lasso = R.lasso.eps = R.rf = R.rf.eps = vector(mode = "list", length = B)
  theta.all = theta.all.eps = matrix(NA, nrow = B, ncol = dim(X.all)[2])
  for (b in 1:B) {
    ##### Split data training-validation sets #####
    ind.train <- sample.int(length(H), size = 0.7 * length(H), replace = F)
    X_train <- X.all[ind.train, ]
    X_test <- X.all[-ind.train, ]
    H_train <- H[ind.train]
    H_test <- H[-ind.train]
    Y_train <- 2 * (H_train > u) - 1
    Y_test <- 2 * (H_test > u) - 1
    Y_train.eps <- 2 * (H_train > v) - 1
    Y_test.eps <- 2 * (H_test > v) - 1
    ##### Linear classifier with mass ######
    theta <- linear.classifier(X = X_train[, c(1, 2)], thresh = u,
                               H = H_train, initials = reslm0, epsilon = 0)$theta
    theta.all[b, ] <- theta
    glin <- 2 * (as.vector(theta %*% t(X_test[, c(1, 2)])) > u) - 1
    R.lin[[b]] <- empirical.risk(Y = Y_test, g = glin, epsilon = 0)
    ##### Linear classifier without mass ######
    theta.eps <- linear.classifier(X = X_train[, c(1, 2)], thresh = u,
                                   H = H_train, initials = reslm, epsilon = eps)$theta
    theta.all.eps[b, ] <- theta.eps
    glineps <- 2 * (as.vector(theta.eps %*% t(X_test[, c(1, 2)])) > v) - 1
    R.lin.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps,
                                     g = glin, g.eps = glineps, epsilon = eps)
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
    R.tree.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps,
                                      g = gtree, g.eps = gtreeeps, epsilon = eps)
    #### SVM linear kernel with mass #####
    w <- which(Y_train == 1)
    X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
    Y_train1 <- c(Y_train[w], Y_train[1:length(w)])
    svmclass <- svm(y ~ ., data = data.frame(x = X_train1, y = Y_train1),
                    type = "C-classification", kernel = "linear")
    gsvm <- as.numeric(as.character(predict(svmclass, 
                                            newdata = data.frame(x = X_test, y = Y_test))))
    R.svml[[b]] <- empirical.risk(Y = Y_test, g = gsvm, epsilon = 0)
    #### SVM linear kernel without mass #####
    w <- which(Y_train.eps == 1)
    X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
    Y_train.eps1 <- c(Y_train.eps[w], Y_train.eps[1:length(w)])
    svmclass.eps <- svm(y ~ ., data = data.frame(x = X_train1, y = Y_train.eps1),
                        type = "C-classification", kernel = "linear")
    gsvmeps <- as.numeric(as.character(predict(svmclass.eps, 
                                               newdata = data.frame(x = X_test, y = Y_test.eps))))
    R.svml.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps,
                                      g = gsvm, g.eps = gsvmeps, epsilon = eps)
    #### Random forest with mass #####
    forest <- randomForest(x = X_train, y = as.factor(Y_train))
    grf <- as.numeric(as.character(predict(forest, 
                                           newdata = cbind(X_test, Y_test))))
    R.rf[[b]] <- empirical.risk(Y = Y_test, g = grf, epsilon = 0)
    #### Random forest without mass #####
    forest.eps <- randomForest(x = X_train, y = as.factor(Y_train.eps))
    grfeps <- as.numeric(as.character(predict(forest.eps, 
                                              newdata = cbind(X_test, Y_test.eps))))
    R.rf.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps,
                                    g = grf, g.eps = grfeps, epsilon = eps)
    #### Logistic regression with mass #####
    w <- which(Y_train == 1)
    X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
    Y_train1 <- c(Y_train[w], Y_train[1:length(w)])
    fraction <- table(Y_train1)/length(Y_train1)
    weights <- 1 - fraction[as.character(Y_train1)]
    m_train1 <- apply(X_train1, 2, mean)
    sd_train1 <- apply(X_train1, 2, sd)
    cv.lasso <- cv.glmnet(x = scale(as.matrix(X_train1)), y = as.factor(Y_train1),
                          weights = weights, family = "binomial")
    glasso <- as.numeric(predict(cv.lasso, scale(as.matrix(X_test),
                                                 center = m_train1, scale = sd_train1), 
                                 type = "class"))
    R.lasso[[b]] <- empirical.risk(Y = Y_test, g = glasso, epsilon = 0)
    #### Logistic regression without mass #####
    w <- which(Y_train.eps == 1)
    X_train1 <- rbind(X_train[w, ], X_train[1:length(w), ])
    Y_train1 <- c(Y_train.eps[w], Y_train.eps[1:length(w)])
    fraction <- table(Y_train1)/length(Y_train1)
    weights <- 1 - fraction[as.character(Y_train1)]
    m_train1 <- apply(X_train1, 2, mean)
    sd_train1 <- apply(X_train1, 2, sd)
    cv.lasso.eps <- cv.glmnet(x = scale(as.matrix(X_train1)), y = as.factor(Y_train1),
                              weights = weights, family = "binomial")
    glassoeps <- as.numeric(predict(cv.lasso.eps, 
                                    scale(as.matrix(X_test), center = m_train1, scale = sd_train1), 
                                    type = "class"))
    R.lasso.eps[[b]] <- empirical.risk(Y = Y_test, Y.eps = Y_test.eps,
                                       g = glasso, g.eps = glassoeps, epsilon = eps)
  }
  ## Second row of Figure 3
  boxplot(cbind(unlist(lapply(R.lin.eps, `[[`, 1)), 
                unlist(lapply(R.tree.eps, `[[`, 1)), 
                unlist(lapply(R.svml.eps, `[[`, 1)),
                unlist(lapply(R.lasso.eps, `[[`, 1)),
                unlist(lapply(R.rf.eps, `[[`, 1))), 
          ylab = "Risk", cex.lab = 2, cex.axis = 2, cex.main = 2, cex.sub = 2, 
          cex = 1.5, ylim = c(0, 1), xaxt = "n", col = "#005AB5", 
          main = bquote(~epsilon[.(k)] == .(eps) ~ u ~ "&" ~ n[v[.(k)]] == .(sum(H_test > v))))
  axis(1, at = c(1, 2, 3, 4, 5), cex = 1.5, lwd = 2, 
       labels = c("Linear", "Tree", "SVM", "Lasso", "RF"), cex.axis = 2)
  ##--
  k <- k + 1
}

###---- Codes for Figure 2 and Table 3, Section 3
set.seed(123)
nsim <- 1e4
alpha1 <- 3
alpha2 <- 2
sigma <- 1
X1 <- sigma*1 / (runif(nsim)^(1/alpha1))
P <- 1 / (runif(nsim)^(1/alpha2))
H <- X1 + P
X2 <- 1/(runif(nsim)^(1/alpha2))
X3 <- rexp(nsim,1)
X4 <- rexp(nsim,2)
u <- quantile(H,probs=0.97)
eps <- 0.7 
eps_u <- u*eps
##---- Code to reproduce Figure 2 Section 3
par(mfrow=c(1,2), mar=c(4.3,5.1,2.1,2.1))
plot(X1,H,pch=20,col='grey', cex.lab=2, cex.axis=2, cex.main=2, cex.sub=2,cex=2,
     xlab='f(X)', ylab="f(X)+N",xlim=range(X1,H),ylim=range(X1,H))
points(X1[H>eps_u & X1>eps_u], H[H>eps_u & X1>eps_u], pch=20,cex=2)
abline(h=u,col='blue',lwd=2,lty='dotted')
abline(v=u,col='blue',lwd=2,lty='dotted')
# ZOOM
plot(X1[H>eps_u & X1>eps_u], H[H>eps_u & X1>eps_u], pch=20,
     cex.lab=2, cex.axis=2, cex.main=2,
     cex.sub=2,cex=2,xlab="f(X)", ylab="f(X)+N",
     xlim=range(X1[H>eps_u & X1>eps_u], H[H>eps_u & X1>eps_u]),
     ylim=range(X1[H>eps_u & X1>eps_u], H[H>eps_u & X1>eps_u]))
abline(h=u,col='blue',lwd=2,lty='dotted')
abline(v=u,col='blue',lwd=2,lty='dotted')
##--

X.all <- cbind(X1,X2,X3,X4)
##### Split data training-validation sets #####
ind.train <- sample.int(length(H), size = 0.7*length(H), replace=F)
X_train <- X.all[ind.train,]
X_test <- X.all[-ind.train,]
H_train <- H[ind.train]
H_test <- H[-ind.train]
Y_train <- 2*(H_train > u) - 1
Y_test <- 2*(H_test > u) - 1
Y_train.eps <- 2*(H_train > eps_u) - 1
Y_test.eps <- 2*(H_test > eps_u) - 1
##### Regression trees with mass ######
library(rpart)
treeclass <- rpart(y~., data=data.frame(x=X_train, y = as.factor(Y_train)), 
                   method = "class")
gtree <- as.numeric(as.character(predict(treeclass, 
                                         newdata = data.frame(x = X_test, y = Y_test),
                                         type="class")))
##### Regression trees without mass ######
treeclass.eps <- rpart(y~., data=data.frame(x=X_train, y = as.factor(Y_train.eps)),
                       method = "class")
gtreeeps <- as.numeric(as.character(predict(treeclass.eps, 
                                            newdata = data.frame(x = X_test, y = Y_test.eps),
                                            type = 'class')))
##---- Code to reproduce Table 3 Section 3
table(gtree,Y_test)
table(gtreeeps,Y_test.eps)
##--