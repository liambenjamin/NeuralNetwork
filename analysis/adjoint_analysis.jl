# set directory to file location
cd("Desktop/NN_Implementation/training/opt_ntwks/")

include("../../src/neurons.jl")
include("../../src/network.jl")
include("../../src/partition.jl")
include("../../src/penalties.jl")
include("../../src/helpers.jl")

using JLD, Plots, SparseArrays, LinearAlgebra


function ntwk_summary(loss :: Module, penalty :: Module)
    info_layer = 5
    n_train = 4000
    N = 1
    num_ntwks = length(1:N)
    F = zeros(num_ntwks)
    Λ = zeros(50, num_ntwks)

    for i=1:num_ntwks
        ntwk = load("layer_5/toy_adjoint_5_2.jld", "ntwk")
        for j=1:n_train
            # find loss
            U, label = randn(1,10), zeros(ntwk.results)
            if sum(U[:,info_layer]) > 0
                label[1] = 1.0
            else
                label[2] = 1.0
            end
            U = collect(Iterators.flatten(U))
            U = vcat(U, zeros(ntwk.hid_dim))
            X = Network.evaluate(ntwk, U)
            λ = Network.adjoint(ntwk, X, label, loss)
            F[i] += loss.evaluate(label, X, ntwk.results)
            Λ[:,i] += λ[16:end-2*ntwk.results]
        end
    end
    return F ./ n_train, Λ ./ n_train
end

lsmod = loss.crossEntropy
pnmod = penalty.var_phi
F, Λ = ntwk_summary(lsmod, pnmod)
F_co, Λ_co = ntwk_summary(lsmod, pnmod)
L = map(i -> reshape(Λ[:,i],5,10), 1:size(Λ,2))
# remove networks that did not converge to minimizer
ind = findall(x -> x < 0.54, F)
F = F[ind]
Λ = L[ind]
