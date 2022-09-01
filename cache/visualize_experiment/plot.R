library(readr)
library(tidyverse)
library(grid)
library(pBrackets)

input <- read_csv("input.csv")
output <- read_csv("output.csv")
sample <- read_csv("pred_sample.csv")
true_sample <- read_csv("full_true_sample.csv")

# past observations
history = data.frame(y = input$`0`, x = -3:0)
# future (true) obsevrations
output = output[3:4]$`y`
output$x = 1:2
# future (predicted) obsevrations
output$sample = I(as.matrix(sample))
colnames(output) = c("y", "x", "sample")

dens <- do.call(rbind, lapply(split(output, output$section), function(x) {
  d = density(x$sample, n = 200)
  res = data.frame(x = x$x - d$y * 0.45, y = d$x)
  res = res[order(res$y), ]
  res$type <- rep(c("empirical"), each=200)
  res
}))
dens$section <- rep(levels(output$section), each=200)

true_sample = data.frame(x = -4:6, y=I(t(as.matrix(true_sample))[8:18, ]))
true_sample$section = cut(true_sample$x, breaks)
dens_true <- do.call(rbind, lapply(split(true_sample[6:11, ], true_sample[6:11, ]$section), function(x) {
  d = density(x$y, n = 200)
  res = data.frame(x = x$x - d$y * 0.45, y = d$x)
  res = res[order(res$y), ]
  res$type <- rep(c("true"), each=200)
  res
}))
dens_true$section <- rep(levels(output$section), each=200)

est_sd = true_sample$y[1, ]
dens_other <- do.call(rbind, lapply(split(true_sample[6:11, ], true_sample[6:11, ]$section), function(x) {
  s = rnorm(1000, mean(x$y), est_sd)
  d = density(s, n = 200, from=-0.5, to=1.5)
  res = data.frame(x = x$x - d$y * 0.45, y = d$x)
  res = res[order(res$y), ]
  res$type <- rep(c("other"), each=200)
  res
}))
dens_other$section <- rep(levels(output$section), each=200)

output$y = output$y + rnorm(length(output$y), 0, 0.05)
history$y = history$y + rnorm(length(history$y), 0, 0.05)

plt = ggplot(data.frame(x=true_sample$x, y=true_sample$y[, 1]), aes(x, y, color = 'Train Data')) + 
  geom_line(lwd = 0.5, alpha = 0.2)
for (i in 2:90) {
  plt = plt + geom_line(data=data.frame(x=true_sample$x, 
                                        y=true_sample$y[, i]) + rnorm(length(output$y), 0, 0.05), 
                        aes(x, y, color='Train Data'), lwd = 0.5, alpha = 0.2)
}
plt = plt + geom_line(data=rbind(output[, 1:2], history), aes(x, y, color='Test Sample'), lwd = 1.1) + 
  geom_point(data=rbind(output[, 1:2], history), aes(x, y), color='black') +
  geom_path(data=dens, aes(x, y, group=interaction(section,type), color="(Our) Predicted Density"), lwd=1, alpha=0.5) +
  geom_path(data=dens_true, aes(x, y, group=interaction(section,type), color="(True) Density"), lwd=1, alpha=0.5) + 
  geom_path(data=dens_other, aes(x, y, group=interaction(section,type), color="(Other) Predicted Density"), lwd=1, alpha=0.5) +
  theme_bw() +
  geom_vline(xintercept=breaks, lty=2) +
  scale_colour_manual(name = "", values = c("(Our) Predicted Density" = "darkgreen",
                                            "(True) Density" = "darkorange",
                                            "(Other) Predicted Density" = "darkred",
                                            "Train Data" = "#999999",
                                            "Test Sample" = "darkblue")) +
  guides(color=guide_legend(nrow=2,byrow=TRUE)) +
  xlab("Time") + 
  ylab("Value") +
  theme(legend.position="bottom", text = element_text(size = 13)) + 
  scale_x_continuous(breaks=seq(-4,6,1))
print(plt)

# geom_line(data=history, aes(x,y, color='Model Input'), lwd = 1.1, ) +
#   geom_point(data=history, aes(x,y), color='black') +
