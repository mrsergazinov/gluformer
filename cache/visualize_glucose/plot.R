library(readr)
library(tidyverse)
library(grid)
library(ggpubr)
library(pBrackets)
library(reticulate)
np <- import("numpy")

# load data
input <- np$load("input.npy")
true <- np$load("true.npy")
pred_mean_infmixt <- np$load("pred_mean_infmixt.npy")
pred_var_infmixt <- np$load("pred_var_infmixt.npy")
pred_mean_norm <- np$load("pred_mean_norm.npy")
pred_var_norm <- np$load("pred_var_norm.npy")

dim(pred_var_infmixt) = c(dim(pred_var_infmixt)[1],
                          1,
                          dim(pred_var_infmixt)[2])
plts = list()
for (i in c(1,5,9)) {
  # past and future / predicted observations
  history = data.frame(y = input[i,1:12], x = -11:0)
  infmixt = data.frame(y = true[i,, 1], 
                       x = 1:12, 
                       pred = rowMeans(pred_mean_infmixt[i,,]),
                       muhat_infmixt = I(as.matrix(pred_mean_infmixt[i,,])),
                       varhat_infmixt = I(as.matrix(pred_var_infmixt[i,rep(1, 12),])))
  normal = data.frame(y = true[i,, 1], 
                      x = 1:12, 
                      pred = pred_mean_norm[i,,],
                      muhat_norm = I(as.matrix(pred_mean_norm[i,,])))
  
  # infmixt: estimate density across temporal dimension
  dens_infmixt <- do.call(rbind, lapply(split(infmixt, infmixt$x), function(x) {
    vals = c()
    for (i in 1:length(x$muhat)){
      vals = c(vals, rnorm(100, x$muhat, sqrt(x$varhat)))
    }
    d = density(vals, n=100)
    res = data.frame(x = x$x - 70*d$y, y = d$x)
    res = res[order(res$y), ]
    res
  }))
  dens_infmixt$section <- rep(infmixt$x, each=100)
  
  xs = do.call(rbind, 
          lapply(split(dens_infmixt, dens_infmixt$section), 
                 function(x) {
                   x$y
                 }
  ))
  normal$xs = I(xs)
  
  # normal: estimate density across temporal dimension
  dens_normal <- do.call(rbind, lapply(split(normal, normal$x), function(x) {
    d = dnorm(x$xs, x$muhat_norm, sqrt(pred_var_norm))
    res = data.frame(x = as.vector(x$x - 100*d), 
                     y = as.vector(x$xs))
    res = res[order(res$y), ]
    res
  }))
  dens_normal$section <- rep(normal$x, each=100)
  
  plt = ggplot()
  plt = plt + geom_line(data=rbind(infmixt[, 1:2], history), aes(x, y, color='Test Sample'), lwd = 1.1) + 
    geom_point(data=rbind(infmixt[, 1:2], history), aes(x, y, color='Test Sample')) +
    # geom_line(data=infmixt, aes(x, pred, color='(Mixture) Predicted Density'), lwd = 1.1) + 
    # geom_point(data=infmixt, aes(x, pred, color='(Mixture) Predicted Density')) +
    # geom_line(data=normal, aes(x, pred, color='(Gaussian) Predicted Density'), lwd = 1.1) + 
    # geom_point(data=normal, aes(x, pred, color='(Gaussian) Predicted Density')) +
    geom_path(data=dens_infmixt, aes(x, y, group=interaction(section), color="(Mixture) Predicted Density"), lwd=1, alpha=0.5) +
    geom_path(data=dens_normal, aes(x, y, group=interaction(section), color="(Gaussian) Predicted Density"), lwd=1, alpha=0.5) + 
    theme_bw() +
    geom_vline(xintercept=1:12, lty=2) +
    scale_colour_manual(name = "", values = c("(Mixture) Predicted Density" = "darkgreen",
                                              "(Gaussian) Predicted Density" = "red",
                                              "Train Data" = "#999999",
                                              "Test Sample" = "blue"
                                              )) +
    guides(color=guide_legend(nrow=2,byrow=TRUE)) +
    xlab("Time") + 
    theme(legend.position="bottom", text = element_text(size = 13))
  if (i == 1) {
    plt = plt + ylab("Value")
  } else {
    plt = plt + ylab(" ")
  }
  plts[[i]] = plt
}
ggarrange(plts[[1]], plts[[9]], ncol=2, common.legend = TRUE, legend="bottom")
