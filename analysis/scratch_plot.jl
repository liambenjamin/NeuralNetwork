include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/gru.jl")
include("../src/lstm.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")
include("plot.jl")

#=
using FileIO, DataFrames, CSV

# Load Networks
adj_ntwk = load("toy/rnn_adjoint_20.jld2", "ntwk")
co_ntwk1 = load("toy/rnn_coadjoint_1e1_20.jld2", "ntwk")
co_ntwk2 = load("toy/rnn_coadjoint_1e2_20.jld2", "ntwk")
co_ntwk3 = load("toy/rnn_coadjoint_1e3_20.jld2", "ntwk")
rand_ntwk = load("toy/rnn_randomInit.jld2", "ntwk")
rand_ntwk2 = load("toy/rnn_randomInit2.jld2", "ntwk")

adj_output = load("toy/rnn_adjoint_output.jld2", "output")
co_output1 = load("toy/rnn_coadjoint_1e1_output.jld2", "output")
co_output2 = load("toy/rnn_coadjoint_1e2_output.jld2", "output")
co_output3 = load("toy/rnn_coadjoint_1e3_output.jld2", "output")

lsmod = loss.binary_crossEntropy
pnmod = penalty.var_phi

F, G, E = Plot.unitsq_interpolation(adj_ntwk, rand_ntwk, rand_ntwk2, co_ntwk1, lsmod, pnmod, "rnn")
α = β = collect(0.0:0.05:1.0)

CSV.write("toy/Fr1.csv",  DataFrame(F), writeheader=false)
CSV.write("toy/Gr1.csv",  DataFrame(G), writeheader=false)
CSV.write("toy/Er1.csv",  DataFrame(E), writeheader=false)
CSV.write("toy/mesh.csv",  DataFrame(hcat(α,β)), writeheader=false)

=#


name = ["p1e0", "p1e1", "p1e2", "p1e3", "p1e4", "p1e5"]

E1, Ep1, Ea1, Eap1 = Plot.unpackNtwkInfo(name, "u1")
E7, Ep7, Ea7, Eap7 = Plot.unpackNtwkInfo(name, "u7")
E14, Ep14, Ea14, Eap14 = Plot.unpackNtwkInfo(name, "u14")
E21, Ep21, Ea21, Eap21 = Plot.unpackNtwkInfo(name, "u21")
E28, Ep28, Ea28, Eap28 = Plot.unpackNtwkInfo(name, "u28")

using Plots
plot(0:5, hcat(E1, Ep1, Ea1, Eap1), labels=["Coadj" "Coadg Pert", "Adj", "Adj Pert"])
