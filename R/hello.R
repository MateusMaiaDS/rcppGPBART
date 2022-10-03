#' @useDynLib rcppGPBART
#' @importFrom Rcpp sourceCpp

# # Loading simple example
# n <- 100
# x_train <- x_test <- matrix(seq(-pi,pi,length.out = n ))
# colnames(x_train) <- colnames(x_test) <- c("x.1")
# num_cut <- 20
# # Cut matrix
# numcut <- num_cut
# xcut <- matrix(NA,ncol = ncol(x_train),nrow = numcut)
#
# # Getting possible x values
# for(j in 1:ncol(x_train)){
#         xs <- stats::quantile(x_train[ , j], type=7,
#                               probs=(0:(numcut+1))/(numcut+1))[-c(1, numcut+2)]
#
#         xcut[,j] <-xs
# }
# x_cut <- xcut
#
# y_train <- c(sin(x_train))

# test_grow_tree(x_train = x_train,x_test = x_test,y_train = y_train,x_cut = x_cut,alpha = 0.95,beta = 2,node_min_size = 1,tau = 10,tau_mu = 10)

# 2+2
