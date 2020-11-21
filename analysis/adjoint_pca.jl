include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/gru.jl")
include("../src/lstm.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")

using FileIO, CSV, DataFrames, Statistics, MultivariateStats, LinearAlgebra, SparseArrays, Main.Neurons, MLDatasets, Random, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU


# load network, loss and penalty
printstyled("Loading Network:", color=:yellow)
ntwk = load("mnist3/rnn_coadjoint_1e2_5.jld2", "ntwk")
lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
partition = load("mnist3/rnn_partition.jld2", "partition")
printstyled("\t\t\tCOMPLETE\n", color=:yellow)


printstyled("Loading Data:\n", color=:yellow)
test_x, test_y = Helpers.load_mnist_test()
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
    dir = Helpers.principal_direction(grad)
    U_pert = Helpers.perturb_dataset(test_x, pos, dir, ntwk.seq_length, ϵ)
    # Compute test error/statistics over perturbed dataset
    printstyled("Testing on Unperturbed Dataset:\n")
    org_F, org_G, org_Λ, org_error = Helpers.mnist_test_error(ntwk, test_x, test_y, lsmod, pnmod, "rnn")
    printstyled("Testing on Perturbed Dataset:\n\n")
    pert_F, pert_G, pert_Λ, pert_error = Helpers.mnist_test_error(ntwk, U_pert, test_y, lsmod, pnmod, "rnn")
    # package and save
    org = DataFrame(F=org_F, G=org_G, E=org_error)
    pert = DataFrame(F=pert_F, G=pert_G, E=pert_error)
    f_org = string("../analysis/adjoint/mnist_rnn_$(method)_1e2_org_u$(pos)_$(name)_.jld2")
    f_pert = string("../analysis/adjoint/mnist_rnn_$(method)_1e2_pert_u$(pos)_$(name)_.jld2")
    save(f_org, "output", org)
    save(f_pert, "output", pert)
    printstyled("\tOriginal Err:\t$(org_error)\n",color=:cyan)
    printstyled("\tOriginal F:\t$(org_F)\n",color=:cyan)
    printstyled("\tOriginal G:\t$(org_G)\n\n",color=:cyan)

    printstyled("\tPerturbed Err:\t$(pert_error)\n",color=:cyan)
    printstyled("\tPerturbed F:\t$(pert_F)\n",color=:cyan)
    printstyled("\tPerturbed G:\t$(pert_G)\n\n",color=:cyan)
end

pos = 1
printstyled("Starting Testing:\n",color=:yellow)
perturb_test(ntwk, lsmod, pnmod, 1500, test_x, test_y, partition, "coadjoint", pos, 5e-5, "p1e5")
printstyled("\tTest 1 Complete\n",color=:yellow)
perturb_test(ntwk, lsmod, pnmod, 1500, test_x, test_y, partition, "coadjoint", pos, 5e-4, "p1e4")
printstyled("\tTest 2 Complete\n",color=:yellow)
perturb_test(ntwk, lsmod, pnmod, 1500, test_x, test_y, partition, "coadjoint", pos, 5e-3, "p1e3")
printstyled("\tTest 3 Complete\n",color=:yellow)
perturb_test(ntwk, lsmod, pnmod, 1500, test_x, test_y, partition, "coadjoint", pos, 5e-2, "p1e2")
printstyled("\tTest 4 Complete\n",color=:yellow)
perturb_test(ntwk, lsmod, pnmod, 1500, test_x, test_y, partition, "coadjoint", pos, 5e-1, "p1e1")
printstyled("\tTest 5 Complete\n",color=:yellow)
perturb_test(ntwk, lsmod, pnmod, 1500, test_x, test_y, partition, "coadjoint", pos, 5e0, "p1e0")
printstyled("\tTest 6 Complete\n",color=:yellow)


# Plotting
#ntwk = load("mnist3/rnn_adjoint_1.jld2", "ntwk")

#f = load("../analysis/adjoint/mnist_rnn_u7_1e0_.jld2", )
