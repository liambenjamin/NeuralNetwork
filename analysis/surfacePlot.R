F_test = as.matrix(read.csv("mnist/F_test_1e1.csv", header=FALSE))
F_train = as.matrix(read.csv("mnist/F_train_1e1.csv", header=FALSE))


#require(ggplot2)
require(plotly)

p <- plot_ly(z = F_test, type = "surface")
p2 <- plot_ly(z = F_train, type = "surface")
p 
p2

fig <- plot_ly(z = ~F_test)
fig <- fig %>% add_surface()

fig
x = seq(0,1,1/20)
y = seq(0,1,1/20)
plot_ly(x = x, y = y, z = F_test, type = "surface")


