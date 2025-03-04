
##### Empirical Risk Estimator
## Description:
# Compute the empirical risk estimator as defined in Eq. (11)
# given the binary outputs of a classifier and the true binary outcomes Y. 
# If epsilon>0, supply both the classifier outputs and the observed events 
# given a second threshold as described in Table 1
# 
## Arguments:
# Y - vector of the observed binary outcomes above threshold u
# Y.eps - vector of the observed binary outcomes above threshold v<u
# g - vector of the predicted binary outcomes from a given classifier for threshold u
# g.eps - vector of the predicted binary outcomes from a given classifier for threshold v<u
# epsilon - single numeric indicating whether the extremal risk should be used (espilon=0) or the conditional version (espilon=v/u>0)
empirical.risk = function(Y, Y.eps, g, g.eps, epsilon = 0){
  
  if (epsilon > 0){
    # if (sum(g.eps >= g) != length(g)){
    #   stop("The classifier g is not level-dependent.")}
    Risk <- sum((g != Y) & (Y.eps == 1) & (g.eps == 1))/
      sum((g == 1 | Y == 1) &
            (Y.eps == 1) & (g.eps == 1))
  }
  else{
    Risk <- sum((g != Y)) / sum(((g == 1) | (Y == 1)))
  }
  return(Risk)
}

##### Test for Equal Risk
## Description:
# Test for the risks of two binary classifiers g1 and g2 to be equal given the
# binary outputs of both classifier and the true binary outcomes Y  based on 
# Proposition 11 (former Proposition 9). If epsilon>0, supply the outputs of
# the two classifiers and the observed events given a second threshold as 
# described in Table 1 -- output is the p-value
# 
## Arguments:
# Y - vector of the observed binary outcomes above threshold u
# Y.eps - vector of the observed binary outcomes above threshold v<u
# g1 - vector of the predicted binary outcomes from a given classifier g1 for
#      threshold u
# g1.eps - vector of the predicted binary outcomes from a given classifier g1 
#          for threshold v<u
# g2 - vector of the predicted binary outcomes from a given classifier g2 for
#      threshold u
# g2.eps - vector of the predicted binary outcomes from a given classifier g2 
#          for threshold v<u
# alternative - specifies the alternative hypothesis, should be "two.sided", 
#               "less" or "greater"
# epsilon - single numeric indicating whether the extremal risk should be used 
#           (espilon=0) or the conditional version (espilon=v/u>0)
test.equal.risk = function(Y, Y.eps, g1, g1.eps, g2, g2.eps, 
                           alternative="two.sided", epsilon = 0) {
  n <- length(Y)
  stopifnot(length(g1) == n)
  stopifnot(length(g2) == n)
  stopifnot(alternative %in% c("two.sided", "less", "greater"))
  if (epsilon > 0) {
    stopifnot(length(Y.eps) == n)
    stopifnot(length(g1.eps) == n)
    stopifnot(length(g2.eps) == n)
    pg1 <- sum((g1 == 1 | Y == 1) & (Y.eps == 1) & (g1.eps == 1))/n
    pg2 <- sum((g2 == 1 | Y == 1) & (Y.eps == 1) & (g2.eps == 1))/n
    reps <- 1/n*sum((g1 != Y) & (g2 != Y) & 
                    (Y.eps == 1) & (g1.eps == 1) & (g2.eps == 1))/min(pg1, pg2)
    peps <- 1/n*sum((g1 == 1 | Y == 1) & (g2 == 1 | Y == 1) & 
                      (Y.eps == 1) & (g1.eps == 1) & (g2.eps == 1))/min(pg1, pg2)
    q12eps <- 1/n*sum((g1 != Y) & (g2 == 1 | Y == 1) & 
                        (Y.eps == 1) & (g1.eps == 1) & (g2.eps == 1))/min(pg1, pg2)
    q21eps <- 1/n*sum((g1 == 1 | Y == 1) & (g2 != Y) & 
                        (Y.eps == 1) & (g1.eps == 1) & (g2.eps == 1))/min(pg1, pg2)
  } else {
    pg1 <- sum((g1 == 1 | Y == 1))/n
    pg2 <- sum((g2 == 1 | Y == 1))/n
    reps <- 1/n*sum((g1 != Y) & (g2 != Y))/min(pg1, pg2)
    peps <- 1/n*sum((g1 == 1 | Y == 1) & (g2 == 1 | Y == 1))/min(pg1, pg2)
    q12eps <- 1/n*sum((g1 != Y) & (g2 == 1 | Y == 1))/min(pg1, pg2)
    q21eps <- 1/n*sum((g1 == 1 | Y == 1) & (g2 != Y))/min(pg1, pg2)
  }
  ceps <- min(pg1, pg2)/c(pg1, pg2)
  R1 <- empirical.risk(Y=Y, Y.eps=Y.eps, g=g1, g.eps=g1.eps, epsilon=epsilon)
  R2 <- empirical.risk(Y=Y, Y.eps=Y.eps, g=g2, g.eps=g2.eps, epsilon=epsilon)
  sigma12 <- sqrt(ceps[1]*ceps[2])*(reps - q21eps*R1 - q12eps*R2 + peps*R1*R2)
  sigmasq <- ceps[1]*R1*(1-R1) - 2*sigma12 + ceps[2]*R2*(1-R2)
  zscore <- (R1 - R2) / sqrt(sigmasq/(n*min(pg1,pg2)))
  if (alternative == "two.sided") {
    p.value <- 2*(1-pnorm(zscore))
  } else if (alternative == "less") {
    p.value <- pnorm(zscore)
  } else if (alternative == "greater") {
    p.value <- 1 - pnorm(zscore)
  }
  return(p.value)
}


##### Optimal linear classifier
## Description:
# Compute the optimal linear classifier as given in Prop. 10 (Append. B) 
# by minimizing the empirical risk (defined by emp.risk.lin).
# If epsilon>0 extremal conditional risk is considered, note that no theoritical results are given in this case
# Initial values must be provided which can be estimated by performing a classical linear regression (e.g. using lm function)
## Arguments:
# X - numeric matrix corresponding to the input data we want to classify
# thresh - single numeric giving the threshold u over which an extreme event is defined
# H - numeric vector corresponding to the latent variable that we wish to predict (as defined in Append. B.1)
# initials - initial values for the parameters of the linear classifier to be optimized over
# epsilon - single numeric indicating whether the extremal risk should be used (espilon=0) or the conditional version (espilon=v/u>0)
linear.classifier <- function(X, thresh, H, initials, epsilon = 0) {
  stopifnot(length(initials) == ncol(X))
  nvar <- ncol(as.matrix(X))
  init <- log(abs(initials))
  res <- optim(par = init, fn = emp.risk.lin, X = X, thresh = thresh,
               epsilon = epsilon, H = H, control = list(parscale = rep(0.01, nvar)),
               method = "SANN")
  return(list(theta = exp(res$par), Risk = res$value))
}


##### Empirical risk estimator for the linear classifier
## Description:
# Internal function, only used by the function linear.classifier
emp.risk.lin <- function(theta, X, thresh, H, epsilon = 0) {
  if (length(theta) != ncol(as.matrix(X)) | length(H) != nrow(as.matrix(X))) {
    stop("H, X and theta must have same length.")
  }
  exp.theta <- exp(theta)
  g_values <- as.vector(t(exp.theta) %*% t(X))
  g <- 2 * (g_values > thresh) - 1
  Y <- 2 * (H > thresh) - 1
  if (epsilon > 0) {
    g.eps <- 2 * (g_values > (epsilon * thresh)) - 1
    Y.eps <- 2 * (H > (epsilon * thresh)) - 1
    Risk <- empirical.risk(Y = Y, Y.eps = Y.eps, g = g, g.eps = g.eps,
                           eps = epsilon)
  } else {
    Risk <- empirical.risk(Y = Y, g = g, eps = epsilon)
  }
  return(Risk)
}

