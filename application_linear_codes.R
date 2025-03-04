#### Codes for the features selection application on the river network
#### See Appendix E for more details
library(rpart)
library(e1071)
library(glmnet)
library(randomForest)
source("functions.R")

###############################
# Import data
Data2 <- graphicalExtremes::danube$data_clustered
Data <- apply(Data2, MARGIN = 2, FUN = function(x) (x - min(x)))
##################################

X.all <- Data[, 2:31]
H <- Data[, 1]
thresh <- quantile(H, probs = 0.85)
sum(H > thresh)
B <- 50  # number of samples for the cross validation
R.lin = R.lin.C = R.tree = R.tree.C = R.svml = R.svml.C = R.lasso = R.lasso.C = 
  R.rf = R.rf.C = vector(mode = "list", length = B)
theta.all = theta.all.C = matrix(NA, nrow = B, ncol = dim(X.all)[2])

# linear classifier, optim only on the subset C
reslin0 <- lm(H ~ X.all)
reslm0 <- reslin0$coefficients[-1]
g_0 <- reslm0 %*% t(X.all)
C0 <- (apply(X.all > thresh, MARGIN = 2, FUN = mean))/mean(H > thresh)
ind <- which(C0 != 0)

## Table 7 in Appendix E
xtable::xtable(as.data.frame(C0))
##---

reslin <- lm(H ~ X.all[, ind])
reslm <- reslin$coefficients[-1]

## Figure 8 in Appendix E
par(mfrow = c(1, 3))
for (i in ind) {
  plot(X.all[, i], H, pch = 20, cex.lab = 2, cex.axis = 2, cex.main = 2,
       cex.sub = 2, cex = 2, ylab = "Station 1", 
       xlab = paste("Station ", i + 1))
  abline(h = thresh, col = "blue", lwd = 2, lty = "dotted")
  abline(v = thresh, col = "blue", lwd = 2, lty = "dotted")
}
##---

u <- thresh
for (b in 1:B) {
  ind.train <- sample.int(length(H), size = 0.7 * length(H), replace = F)
  X_train <- X.all[ind.train, ]
  X_test <- X.all[-ind.train, ]
  H_train <- H[ind.train]
  H_test <- H[-ind.train]
  Y_train <- 2 * (H_train > u) - 1
  Y_test <- 2 * (H_test > u) - 1
  
  ##### Linear classifier with all data ######
  theta <- linear.classifier(X = X_train, thresh = u, H = H_train, initials = reslm0,
                             epsilon = 0)$theta
  theta.all[b, ] <- theta
  glin <- 2 * (as.vector(theta %*% t(X_test)) > u) - 1
  R.lin[[b]] <- empirical.risk(Y = Y_test, g = glin, epsilon = F)
  
  ##### Linear classifier only ci!=0 ######
  theta <- linear.classifier(X = X_train[, ind], thresh = u, H = H_train,
                             initials = reslm, epsilon = 0)$theta
  theta.all.C[b, ] <- theta
  glin.C <- 2 * (as.vector(theta %*% t(X_test[, ind])) > u) - 1
  R.lin.C[[b]] <- empirical.risk(Y = Y_test, g = glin.C, epsilon = F)
  
  ##### Regression trees with all data ######
  treeclass <- rpart(y ~ ., data = data.frame(x = X_train, y = as.factor(Y_train)),
                     method = "class")
  gtree <- as.numeric(as.character(predict(treeclass, 
                                           newdata = data.frame(x = X_test, y = Y_test), 
                                           type = "class")))
  R.tree[[b]] <- empirical.risk(Y = Y_test, g = gtree, epsilon = F)
  
  ##### Regression trees only ci!=0 ######
  treeclass <- rpart(y ~ ., data = data.frame(x = X_train[, ind], y = as.factor(Y_train)),
                     method = "class")
  gtree.C <- as.numeric(as.character(predict(treeclass, 
                                             newdata = data.frame(x = X_test[ , ind], y = Y_test),
                                             type = "class")))
  R.tree.C[[b]] <- empirical.risk(Y = Y_test, g = gtree.C, epsilon = F)
  
  #### SVM linear kernel with all data #####
  svmclass <- svm(y ~ ., data = data.frame(x = X_train, y = Y_train),
                  type = "C-classification", kernel = "linear")
  gsvm <- as.numeric(as.character(predict(svmclass, 
                                          newdata = data.frame(x = X_test, y = Y_test))))
  R.svml[[b]] <- empirical.risk(Y = Y_test, g = gsvm, epsilon = F)
  
  #### SVM linear kernel only ci!=0 #####
  svmclass <- svm(y ~ ., data = data.frame(x = X_train[, ind], y = Y_train),
                  type = "C-classification", kernel = "linear")
  gsvm.C <- as.numeric(as.character(predict(svmclass, 
                                            newdata = data.frame(x = X_test[, ind], y = Y_test))))
  R.svml.C[[b]] <- empirical.risk(Y = Y_test, g = gsvm.C, epsilon = F)
  
  #### Random forest with all data #####
  forest <- randomForest(x = X_train, y = as.factor(Y_train))
  grf <- as.numeric(as.character(predict(forest, newdata = cbind(X_test, Y_test))))
  R.rf[[b]] <- empirical.risk(Y = Y_test, g = grf, epsilon = F)
  
  #### Random forest only ci!=0 #####
  forest <- randomForest(x = X_train[, ind], y = as.factor(Y_train))
  grf.C <- as.numeric(as.character(predict(forest, newdata = cbind(X_test[, ind], Y_test))))
  R.rf.C[[b]] <- empirical.risk(Y = Y_test, g = grf.C, epsilon = F)
  
  #### Logistic regression with all data #####
  fraction <- table(Y_train)/length(Y_train)
  weights <- 1 - fraction[as.character(Y_train)]
  m_train1 <- apply(X_train, 2, mean)
  sd_train1 <- apply(X_train, 2, sd)
  cv.lasso <- cv.glmnet(x = scale(as.matrix(X_train)), y = as.factor(Y_train),
                        weights = weights, family = "binomial")
  glasso <- as.numeric(predict(cv.lasso, 
                               scale(as.matrix(X_test), center = m_train1, scale = sd_train1), 
                               type = "class"))
  R.lasso[[b]] <- empirical.risk(Y = Y_test, g = glasso, epsilon = F)
  
  #### Logistic regression only ci!=0 #####
  fraction <- table(Y_train)/length(Y_train)
  weights <- 1 - fraction[as.character(Y_train)]
  m_train1 <- apply(X_train[, ind], 2, mean)
  sd_train1 <- apply(X_train[, ind], 2, sd)
  cv.lasso <- cv.glmnet(x = scale(as.matrix(X_train[, ind])), y = as.factor(Y_train),
                        weights = weights, family = "binomial")
  glasso.C <- as.numeric(predict(cv.lasso, 
                                 scale(as.matrix(X_test[, ind]), center = m_train1, scale = sd_train1), 
                                 type = "class"))
  R.lasso.C[[b]] <- empirical.risk(Y = Y_test, g = glasso.C, epsilon = F)
}
## Figure 9 from Appendix E
par(mfrow = c(1, 1))
boxplot(cbind(unlist(lapply(R.lin.C, `[[`, 1)), 
              unlist(lapply(R.tree, `[[`, 1)), 
              unlist(lapply(R.svml, `[[`, 1)), 
              unlist(lapply(R.lasso, `[[`, 1)), unlist(lapply(R.rf, `[[`, 1))), 
        ylab = "Risk", cex.lab = 2, cex.axis = 2,
        cex.main = 2, cex.sub = 2, cex = 1.5, ylim = c(0, 0.5), xaxt = "n",
        col = "#DC3220", at = c(1, 4, 7, 10, 13), xlim = c(0.5, 14.5))

boxplot(cbind(unlist(lapply(R.lin.C, `[[`, 1)), 
              unlist(lapply(R.tree.C, `[[`, 1)),
              unlist(lapply(R.svml.C, `[[`, 1)), 
              unlist(lapply(R.lasso.C, `[[`, 1)), 
              unlist(lapply(R.rf.C, `[[`, 1))), 
        ylab = "Risk", cex.lab = 2,
        cex.axis = 2, cex.main = 2, cex.sub = 2, cex = 1.5, ylim = c(0, 1),
        xaxt = "n", add = T, at = c(2, 5, 8, 11, 14), names = F, col = "#3CB371")
axis(1, at = c(1.5, 4.5, 7.5, 10.5, 13.5), cex = 1.5, lwd = 2, 
     labels = c("Linear", "Tree", "SVM", "Lasso", "RF"), cex.axis = 1.5)
##---

## Figure 7 from Appendix E (river map)
library(tidyverse)
library(osmdata)
library(sf)
library(ggmap)

par(mfrow=c(1,1), mar=c(4.3,5.1,2.1,2.1))

riverInfo <- graphicalExtremes::danube$info
rivers <- getbb("Bavaria")%>%
  opq(timeout = 100) %>%
  add_osm_feature(key = "waterway", 
                  value = "river",
                  bbox = c(min(riverInfo$Long),min(riverInfo$Lat),
                           max(riverInfo$Long),max(riverInfo$Lat))) %>%
  osmdata_sf()

plot(riverInfo$Long[-c(23,24,1)],riverInfo$Lat[-c(23,24,1)],col='red',pch=20,cex=2,
     xlab="longitude",ylab="latitude",
     ylim=c(47.4,max(riverInfo$Lat)),cex.lab=2,cex.axis=2,
     xlim=c(min(riverInfo$Long),13.6),
     cex.main=2,cex.sub=2)
plot(rivers$osm_multilines$geometry,add=T,col="#08519c",cex=1.5)
text(labels=c(2:17,19:24,26:31),x=riverInfo$Long[-c(1,18,25)], 
     y=riverInfo$Lat[-c(1,18,25)],col='black',
     font=2,cex=1.5,
     adj=c(0.5,1.3))
text(labels=c(1,18,25),x=riverInfo$Long[c(1,18,25)], 
     y=riverInfo$Lat[c(1,18,25)],col='black',
     font=2,cex=1.5,
     adj=c(-0.4,0.5))
points(riverInfo$Long[c(23,24,1)],riverInfo$Lat[c(23,24,1)],
       bg='darkgreen',pch=24,cex=1.5)
##--
