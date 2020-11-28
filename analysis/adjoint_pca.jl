include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/gru.jl")
include("../src/lstm.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")

using FileIO, CSV, DataFrames, Statistics, MultivariateStats, LinearAlgebra, SparseArrays, Main.Neurons, MLDatasets, Random, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU



#=
N=1500
test_x, test_y = Helpers.load_mnist_test()
x, y = test_x[1:1000], test_y[1:1000,:]
printstyled("\t\t\t\tCOMPLETE\n", color=:yellow)

printstyled("Computing gradient matrix for u₁:", color=:yellow)
grad = Helpers.input_element_grad(ntwk, lsmod, pnmod, x, y, partition, "adjoint", "rnn", 1)
printstyled("\tCOMPLETE\n", color=:yellow)


printstyled("Computing first principal direction:", color=:yellow)
dir = Helpers.principal_direction(grad)
printstyled("\tCOMPLETE\n", color=:yellow)

printstyled("Generating perturbed dataset:", color=:yellow)
ϵ = 1e0
U_pert = Helpers.perturb_dataset(test_x, 1, dir, ntwk.seq_length, ϵ)
printstyled("\t\tCOMPLETE\n\n\n", color=:yellow)


# Compute test error/statistics over perturbed dataset
printstyled("Testing on Unperturbed Dataset:")
org_F, org_G, org_Λ, org_error = Helpers.mnist_test_error(ntwk, test_x, test_y, lsmod, pnmod, "rnn")
printstyled("\t\tCOMPLETE\n\n", color=:yellow)
printstyled("Testing on Perturbed Dataset:")
pert_F, pert_G, pert_Λ, pert_error = Helpers.mnist_test_error(ntwk, U_pert, test_y, lsmod, pnmod, "rnn")
printstyled("\t\tCOMPLETE\n\n", color=:yellow)

printstyled("Results:\n\n",color=:magenta)
printstyled("\tOriginal Err:\t$(org_error)\n",color=:cyan)
printstyled("\tOriginal F:\t$(org_F)\n",color=:cyan)
printstyled("\tOriginal G:\t$(org_G)\n\n",color=:cyan)

printstyled("\tPerturbed Err:\t$(pert_error)\n",color=:cyan)
printstyled("\tPerturbed F:\t$(pert_F)\n",color=:cyan)
printstyled("\tPerturbed G:\t$(pert_G)\n\n",color=:cyan)

=#

"""
N : number of samples for gradient matrix
test_x, test_y : output from load_mnist_test()
"""


function perturb_test(ntwk :: Network.network, loss :: Module, penalty :: Module, N :: Integer, test_x, test_y, partition :: Vector{Vector{Int64}}, method :: String, pos :: Int64, ϵ :: Float64, name :: String)

    x, y = test_x[1:N], test_y[1:N,:]
    grad = Helpers.input_element_grad(ntwk, loss, penalty, x, y, partition, method, "rnn", pos)
    dir = Helpers.principal_direction(grad, 2) # 1 : dominant direction
    U_pert = Helpers.perturb_dataset(test_x, pos, dir, ntwk.seq_length, ϵ)
    # Compute test error/statistics over perturbed dataset
    printstyled("Testing on Unperturbed Dataset:\n")
    org_F, org_G, org_Λ, org_error = Helpers.mnist_test_error(ntwk, test_x, test_y, lsmod, pnmod, "rnn")
    printstyled("Testing on Perturbed Dataset:\n\n")
    pert_F, pert_G, pert_Λ, pert_error = Helpers.mnist_test_error(ntwk, U_pert, test_y, lsmod, pnmod, "rnn")
    # package and save
    org = DataFrame(F=org_F, G=org_G, E=org_error)
    pert = DataFrame(F=pert_F, G=pert_G, E=pert_error)
    f_org = string("adjoint/mnist_rnn_$(method)_org_u$(pos)_$(name)_d2.jld2")
    f_pert = string("adjoint/mnist_rnn_$(method)_pert_u$(pos)_$(name)_d2.jld2")
    save(f_org, "output", org)
    save(f_pert, "output", pert)
    printstyled("\tOriginal Err:\t$(org_error)\n",color=:cyan)
    printstyled("\tOriginal F:\t$(org_F)\n",color=:cyan)
    printstyled("\tOriginal G:\t$(org_G)\n\n",color=:cyan)

    printstyled("\tPerturbed Err:\t$(pert_error)\n",color=:cyan)
    printstyled("\tPerturbed F:\t$(pert_F)\n",color=:cyan)
    printstyled("\tPerturbed G:\t$(pert_G)\n\n",color=:cyan)
end

# load network, loss and penalty
printstyled("Loading Network:", color=:yellow)
method = "adjoint"
ntwk = method == "adjoint" ?
    load("mnist3/rnn_adjoint_5.jld2", "ntwk") :
    load("mnist3/rnn_coadjoint_1e2_5.jld2", "ntwk")
lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
partition = load("mnist3/rnn_partition.jld2", "partition")
printstyled("\t\t\tCOMPLETE\n", color=:yellow)

printstyled("Loading Data:\n", color=:yellow)
test_x, test_y = Helpers.load_mnist_test()

pos = 28
N = 2000 # samples used to generate principal direction
m = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1e0]
names = ["5e4", "1e3", "5e3", "1e2","5e2", "1e1", "2.5e1", "5e1", "7.5e1", "1e0"]

printstyled("Starting Testing:\n",color=:yellow)
for i=1:length(m)
    printstyled("\t Test $(i)/$(length(m))\n",color=:yellow)
    perturb_test(ntwk, lsmod, pnmod, N, test_x, test_y, partition, method, pos, m[i], names[i])
end
printstyled("\n\nTesting Complete.\n\n", color=:yellow)
