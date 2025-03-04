#### simple reproducible codes using the empirical risk
#### estimator function
require(rpart)
source("functions.R")

set.seed(123)

## Simulate data as in Section 3
nsim <- 10000
X1 <- 1/(runif(nsim)^(1/3))
X2 <- 1/(runif(nsim)^(1/2))
N <- 1/(runif(nsim)^(1/2))
H <- X1 + N

## !! Note that if one already have outputs from a classifier
## as in Table 1, there no need to perform the following and
## the user can proceed to line 52

## Compute the two thresholds u and v
u <- quantile(H, probs = 0.97)
eps <- 0.6
v <- u * eps

## Split data between training and testing sets
ii <- sample.int(length(H), size = 0.7 * length(H), replace = F)
Xtrain <- cbind(X1[ii], X2[ii])
Xtest <- cbind(X1[-ii], X2[-ii])
Htrain <- H[ii]
Htest <- H[-ii]

## Linear classifier ~~~~~~~~~~~~~~~~~~~ Train the linear
## classifier with only one threshold (i.e. with mass)
init0 <- lm(H ~ X1 + X2, data = data.frame(H = H, X1 = X1, X2 = X2))$coefficients[2:3]
linclass <- linear.classifier(X = Xtrain, thresh = u, H = Htrain,
    initials = init0, epsilon = 0)$theta
## Compute the predicted binary outcome on the test set from
## all the data
glin <- 2 * (as.vector(linclass %*% t(Xtest)) > u) - 1
# ~~~~~~~~~~~~~~~~~~~ Train the linear classifier with two
# thresholds (i.e. without mass)
glin0 <- init0 %*% t(cbind(X1, X2))
init <- lm(H ~ X1 + X2, data = data.frame(H = H[H > v & glin0 >
    v], X1 = X1[H > v & glin0 > v], X2 = X2[H > v & glin0 > v]))$coefficients[2:3]
linclass.eps <- linear.classifier(X = Xtrain[, c(1, 2)], thresh = u,
    H = Htrain, initials = init, epsilon = eps)$theta
## Compute the predicted binary outcome on the test set from
## the extreme region data
glineps <- 2 * (as.vector(linclass.eps %*% t(Xtest)) > v) - 1
# ~~~~~~~~~~~~~~~~~~~ Compute the true binary outcome on the
# test set from all the data and only with the extreme region
# data
Ytesteps <- 2 * (Htest > v) - 1
Ytest <- 2 * (Htest > u) - 1
# ~~~~~~~~~~~~~~~~~~~

## Estimate the associated extremal conditional risk
empirical.risk(Y = Ytest, Y.eps = Ytesteps, g = glin, g.eps = glineps,
    epsilon = eps)

## Contingency tables
table(Ytest, glin)  # threshold u
table(Ytesteps, glineps)  # threshold v 

## Instead of the linear classifier, one can consider another
## classical classifier, for instance decision trees (again we
## compute by hand the ingredients needed as in Table 1):

## Comparison with regression tree ~~~~~~~~~~~~~~~~~~~ Train
## the tree classifier with mass
Ytrain <- 2 * (Htrain > u) - 1
treeclass <- rpart(y ~ ., data = data.frame(x = Xtrain, y = as.factor(Ytrain)),
    method = "class")
## Compute the predicted binary outcome on the test set from
## all the data
gtree <- as.numeric(as.character(predict(treeclass, newdata = data.frame(x = Xtest,
    y = Ytest), type = "class")))
# ~~~~~~~~~~~~~~~~~~~ Train the tree classifier without mass
Ytraineps <- 2 * (Htrain > v) - 1
treeclasseps <- rpart(y ~ ., data = data.frame(x = Xtrain, y = as.factor(Ytraineps)),
    method = "class")
## Compute the predicted binary outcome on the test set from
## the extreme region data
gtreeeps <- as.numeric(as.character(predict(treeclasseps, newdata = data.frame(x = Xtest,
    y = Ytesteps), type = "class")))
# ~~~~~~~~~~~~~~~~~~~ Compute the associated risk
empirical.risk(Y = Ytest, Y.eps = Ytesteps, g = gtree, g.eps = gtreeeps,
    epsilon = eps)

## Contingency tables
table(Ytest, gtree)  # threshold u
table(Ytesteps, gtreeeps)  # threshold v 
