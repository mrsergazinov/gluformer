library(readr)
library(tidyverse)
library(grid)
library(ggpubr)
library(pBrackets)
library(reticulate)
library(RColorBrewer)
np <- import("numpy")

# load data
calibration <- np$load("calibration_infmixt.npy")

probs = seq(0, 1, 0.1)
output = matrix(nrow=11, ncol=12)
# output[,1] = probs
for (i in  1:dim(calibration)[1]) {
  for (j in 1:length(probs)){
    output[j, i] = mean(calibration[i,] <= probs[j])
  }
}
df = list()
df[["Observed confidence"]] = as.vector(output)
df[["Expected confidence"]] = rep(probs, 12)
df[["L"]] = as.factor(rep(1:12, each=11))
df = data.frame(df)
  
plt <- ggplot(data=df, aes(x=Expected.confidence,
                           y=Observed.confidence,
                           color=L)) + 
  geom_path() + 
  geom_point() + 
  scale_colour_brewer(palette="Paired") +
  xlab("Expected confidence") + ylab("Observed confidence") +
  theme_bw() +
  theme(legend.position="bottom") + 
  guides(fill=guide_legend(nrow=2,byrow=TRUE))
print(plt)

# def plot_calibration(calibration_path, calibration):
#     calibration_matrix = np.empty((11, 13))
#     probs = np.linspace(0, 1, 11)
#     for i in range(12):
#         for j in range(len(probs)):
#             calibration_matrix[j, i+1] = np.mean(np.array(calibration[i]) <= probs[j])
#     calibration_matrix[:, 0] = probs
#     calibration_data = pd.DataFrame(calibration_matrix)
#     calibration_data.columns = ["Expected Confidence"] + [str(i * 5) + " minutes" for i in range(1, 13)]
#     calibration_data = calibration_data.melt(id_vars=["Expected Confidence"], var_name="Time", value_name="Observed Confidence")

#     sns.set_theme()
#     sns.set_context("paper")
#     # Initialize a grid of plots with an Axes for each walk
#     grid = sns.FacetGrid(calibration_data, col="Time", hue="Time", palette="tab20c",
#                         col_wrap=6, height=2)
#     # Draw a line plot to show the trajectory of each random walk
#     grid.map(plt.plot, "Expected Confidence", "Observed Confidence", marker="o")
#     # Adjust the tick positions and labels
#     grid.set(xticks=[0, 0.2, 0.4, 0.6, 0.8, 1], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
#             xlim=(-.1, 1.1), ylim=(-.1, 1.1))
#     # PLot diagonal 45 lines
#     for ax in grid.axes.flat:
#         x = np.linspace(0, 1, 11)
#         y = x
#         ax.plot(x, y, linestyle=':', color='gray')
#     # Adjust the arrangement of the plots
#     grid.fig.tight_layout(w_pad=1)
#     plt.savefig(calibration_path, dpi=300)