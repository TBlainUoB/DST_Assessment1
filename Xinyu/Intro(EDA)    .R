# general visualisation
library('ggplot2') # visualisation
library('scales') # visualisation
library('grid') # visualisation
library('ggthemes') # visualisation
library('gridExtra') # visualisation
library('RColorBrewer') # visualisation
library('corrplot') # visualisation

# general data manipulation
library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('rlang') # data manipulation

# specific visualisation
library('alluvial') # visualisation
#library('ggfortify') # visualisation
library('ggrepel') # visualisation
library('ggridges') # visualisation
library('VIM') # NAs
library('plotly') # interactive
library('ggforce') # visualisation

# modelling
library('xgboost') # modelling
library('caret') # modelling
library('MLmetrics') # gini metric

# Define multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#


multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}

# function to extract binomial confidence levels
get_binCI <- function(x,n) as.list(setNames(binom.test(x,n)$conf.int, c("lwr", "upr")))

train <- as.tibble(fread('/Users/xinyu/Downloads/Data/train.csv', na.strings=c("-1","-1.0")))
test <- as.tibble(fread('/Users/xinyu/Downloads/Data/test.csv', na.strings=c("-1","-1.0")))
sample_submit <- as.tibble(fread('/Users/xinyu/Downloads/Data/sample_submission.csv'))

summary(train)
glimpse(train)
summary(test)
glimpse(test)
sum(is.na(train))
sum(is.na(test))


#Reformating features
train <- train %>%
  mutate_at(vars(ends_with("cat")), funs(factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical)) %>%
  mutate(target = as.factor(target))
test <- test %>%
  mutate_at(vars(ends_with("cat")), funs(factor)) %>%
  mutate_at(vars(ends_with("bin")), funs(as.logical))


#Combining data frames
combine <- bind_rows(train %>% mutate(dset = "train"), 
                     test %>% mutate(dset = "test",
                                     target = NA))
combine <- combine %>% mutate(dset = factor(dset))

#Binary features
p1 <- train %>%
  ggplot(aes(ps_ind_06_bin, fill = ps_ind_06_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p2 <- train %>%
  ggplot(aes(ps_ind_07_bin, fill = ps_ind_07_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p3 <- train %>%
  ggplot(aes(ps_ind_08_bin, fill = ps_ind_08_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p4 <- train %>%
  ggplot(aes(ps_ind_09_bin, fill = ps_ind_09_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p5 <- train %>%
  ggplot(aes(ps_ind_10_bin, fill = ps_ind_10_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p6 <- train %>%
  ggplot(aes(ps_ind_11_bin, fill = ps_ind_11_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p7 <- train %>%
  ggplot(aes(ps_ind_12_bin, fill = ps_ind_12_bin)) +
  geom_bar() +
  theme(legend.position = "none")

p8 <- train %>%
  ggplot(aes(ps_ind_13_bin, fill = ps_ind_13_bin)) +
  geom_bar() +
  theme(legend.position = "none")

layout <- matrix(c(1,2,3,4,5,6,7,8),2,4,byrow=TRUE)
multiplot(p1, p2, p3, p4, p5, p6, p7, p8, layout=layout)






