library("fdacluster")
library(data.table)
library(magrittr)
library(openxlsx)

setwd('/Users/apollinaria45/Documents/PhD/InVitro_Code/ScanOFC_ALL/Code_OOP/scanofc')
x1 <- c(0.5,  3 ,  6 ,  9 ,  12 ,  15 , 18 , 21)
x1 <- t(as.matrix(replicate(300, x1)))
y1 <- as.matrix(read.csv(paste0(getwd(),"/sim_means.csv"))[,2:9])
x1 <- x1[1:20,]
y1 <- y1[1:20,]
out1 <- fdakmeans(x1, y1, n_clusters = 4, 
                  centroid_type = "medoid",
                  warping_class = "shift",
                  add_silhouettes = FALSE,
                  distance_relative_tolerance=0.2,
                  warping_options = c(0.1, 0.1)
                  )
plot(out1, type = "amplitude")
plot(out1, type = "phase")
# True clusters: array([0, 2, 1, 1, 1, 2, 3, 3, 2, 0, 3, 1, 1, 0, 0, 0, 2, 2, 3, 3])
write.xlsx(data.frame(out1$memberships), file = "fda_res_300_2.xlsx", sheetName = "array")
write.xlsx(data.frame(out1$center_curves), file = "fda_res_300_2_centroids.xlsx", sheetName = "array")
write.xlsx(data.frame(out1$distances_to_center), file = "fda_res_300_2_dist.xlsx", sheetName = "array")
