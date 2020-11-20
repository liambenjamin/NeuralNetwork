include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/gru.jl")
include("../src/lstm.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")
#include("plot.jl")

using FileIO, Statistics, MultivariateStats, LinearAlgebra, SparseArrays, Main.Neurons, MLDatasets, Random, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU

# load network, loss and penalty
printstyled("Loading Network:", color=:yellow)
ntwk = load("mnist3/rnn_adjoint_1.jld2", "ntwk")
lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
partition = load("mnist3/rnn_partition.jld2", "partition")
printstyled("\tCOMPLETE\n", color=:yellow)
printstyled("Loading Test Data:", color=:yellow)
test_x, test_y = Helpers.load_mnist_test()
printstyled("\tCOMPLETE\n\n", color=:yellow)

# MNIST
N = 500
printstyled("Computing gradient matrices over $(N) test features:\n\n", color=:yellow)
printstyled("\t∇uᵢ where i ∈ {1,7,14,21,28}", color=:yellow)
test_x, test_y = test_x[1:500], test_y[1:500,:] # trim dataset to first 100 samples
dU1 = Helpers.input_element_grad(ntwk, lsmod, pnmod, test_x, test_y, partition, "adjoint", "rnn", 1)
dU7 = Helpers.input_element_grad(ntwk, lsmod, pnmod, test_x, test_y, partition, "adjoint", "rnn", 7)
dU14 = Helpers.input_element_grad(ntwk, lsmod, pnmod, test_x, test_y, partition, "adjoint", "rnn", 14)
dU21 = Helpers.input_element_grad(ntwk, lsmod, pnmod, test_x, test_y, partition, "adjoint", "rnn", 21)
dU28 = Helpers.input_element_grad(ntwk, lsmod, pnmod, test_x, test_y, partition, "adjoint", "rnn", 28)
printstyled("\t\tCOMPLETE\n\n",color=:yellow)
printstyled("Performing PCA on gradient matrices:")
P1 = fit(PCA, dU1, method=:cov)
P7 = fit(PCA, dU7, method=:cov)
P14 = fit(PCA, dU14, method=:cov)
P21 = fit(PCA, dU21, method=:cov)
P28 = fit(PCA, dU28, method=:cov)
printstyled("\tCOMPLETE\n\n", color=:yellow)

printstyled("Results for PCA on ∇u1:\n", color=:magenta)
printstyled("\t\tout dimension:\t$(outdim(P1))\n",color=:magenta)
printstyled("\t\tvariance of principal components:\t$(principalvars(P1))\n\n\n",color=:magenta)

printstyled("Results for PCA on ∇u7:\n", color=:magenta)
printstyled("\t\tout dimension:\t$(outdim(P7))\n",color=:magenta)
printstyled("\t\tvariance of principal components:\t$(principalvars(P7))\n\n\n",color=:magenta)

printstyled("Results for PCA on ∇u14:\n", color=:magenta)
printstyled("\t\tout dimension:\t$(outdim(P14))\n",color=:magenta)
printstyled("\t\tvariance of principal components:\t$(principalvars(P14))\n\n\n",color=:magenta)

printstyled("Results for PCA on ∇u21:\n", color=:magenta)
printstyled("\t\tout dimension:\t$(outdim(P21))\n",color=:magenta)
printstyled("\t\tvariance of principal components:\t$(principalvars(P21))\n\n\n",color=:magenta)

printstyled("Results for PCA on ∇u28:\n", color=:magenta)
printstyled("\t\tout dimension:\t$(outdim(P28))\n",color=:magenta)
printstyled("\t\tvariance of principal components:\t$(principalvars(P28))\n\n\n",color=:magenta)

function perturb_input(train_x, G :: )

function perturb_test()
