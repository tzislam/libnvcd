pc_avg <- c(41146, 45606, 45964, 34281)
wp_avg <- c(65048, 110444, 146851, 103162)

pc_max <- c(41754, 66170, 46031, 111615, 46544, 147636, 34952, 106228)
pc_min <- c(40082, 62996, 45331, 108674, 44972, 146143, 33444, 99607)

pc_x <- c(5, 10, 20, 30)

m <- matrix(c(
  pc_avg[1], wp_avg[1],
  pc_avg[2], wp_avg[2],
  pc_avg[3], wp_avg[3],
  pc_avg[4], wp_avg[4]
), 
nr=2, 
nc=4);

bx <- barplot(m, beside=T, ylim=c(0, 150000),
        col=c("cyan","red"), 
        names.arg=c("n = 4", "n = 8", "n = 12", "n = 16"),
        xlab="Number of processes", ylab="Messages per second",
        main="Average, min, and max")

legend("topleft", c("producer, consumer","workpool"), pch=20, 
       col=c("cyan","red"), 
       bty="n")

arrows(bx, pc_min, bx, pc_max, angle=90, code=3)