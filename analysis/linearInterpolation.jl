include("../../src/neurons.jl")
include("../../src/network.jl")
include("../../src/partition.jl")
include("../../src/penalties.jl")
include("../../src/helpers.jl")
include("../../src/rnn.jl")

using FileIO, JLD2, Plots, SparseArrays, LinearAlgebra, CSV, DataFrames

"""
Linear Interpolation of Loss Surface
θ = (1-α) × θι + α × θɾ
"""

function linear_interpolation(ntwk_1, ntwk_2, loss :: Module, penalty :: Module, info_layer :: Int64, alpha, ntwk_type :: String)
	L = length(ntwk_1.neurons)
	N = 1000
	ntwk_comb = deepcopy(ntwk_1)
	F = zeros(length(α))
	G = zeros(length(α))

	for i=1:length(α)

		# linear combination of ntwk1 and ntwk2
		for j=1:L
			ntwk_comb.neurons[j].β = ((1 - α[i]) * ntwk_1.neurons[j].β) + (α[i] * ntwk_2.neurons[j].β)
		end
		# compute J(θ) for N samples
		for k=1:N
			U, label = randn(1,10), zeros(ntwk_comb.results)
			if sum(U) > 0
				label[1] = 1.0
			else
				label[2] = 1.0
			end
			U = collect(Iterators.flatten(U))
			U = ntwk_type != "lstm" ?
				vcat(U, zeros(ntwk_comb.hid_dim)) :
				vcat(U, zeros(2*ntwk_comb.hid_dim))
			X = Network.evaluate(ntwk_comb, U)
			Λ = Network.adjoint(ntwk_comb, X, label, loss)
			F[i] += loss.evaluate(label, X, ntwk_comb.results)
			G[i] += penalty.evaluate(Λ, ntwk_comb.features, ntwk_comb.hid_dim, ntwk_comb.seq_length, ntwk_type)
		end
	end
	return F ./ N, G ./ N
end


"""
Loading Trained Networks
"""
# RNNs
rnn_1 = load("rnn/rnn_adjoint_1_50.jld2", "ntwk")
rnn_2 = load("rnn/rnn_adjoint_2_50.jld2", "ntwk")
rnn_3 = load("rnn/rnn_adjoint_3_50.jld2", "ntwk")
rnn_init1 = load("rnn/rnn_init_1.jld2", "ntwk")
rnn_init2 = load("rnn/rnn_init_2.jld2", "ntwk")
rnn_init3 = load("rnn/rnn_init_3.jld2", "ntwk")
# LSTMs
lstm_1 = load("lstm/lstm_adjoint_1_50.jld2", "ntwk")
lstm_2 = load("lstm/lstm_adjoint_2_50.jld2", "ntwk")
lstm_3 = load("lstm/lstm_adjoint_3_50.jld2", "ntwk")
lstm_init1 = load("lstm/lstm_init_1.jld2", "ntwk")
lstm_init2 = load("lstm/lstm_init_2.jld2", "ntwk")
lstm_init3 = load("lstm/lstm_init_3.jld2", "ntwk")
# GRUs
gru_1 = load("gru/gru_adjoint_1_50.jld2", "ntwk")
gru_2 = load("gru/gru_adjoint_2_50.jld2", "ntwk")
gru_3 = load("gru/gru_adjoint_3_50.jld2", "ntwk")
gru_init1 = load("gru/gru_init_1.jld2", "ntwk")
gru_init2 = load("gru/gru_init_2.jld2", "ntwk")
gru_init3 = load("gru/gru_init_3.jld2", "ntwk")


"""
Linear Interpolation Plots
"""
lsmod = loss.crossEntropy
pnmod = penalty.var_phi
info_layer = 4
mesh = 0:0.05:2
α = map(i -> mesh[i], 1:length(mesh))
printstyled("Starting RNN Linear Interpolation...\n", color=:cyan)
# RNNs
F_rnn1, _ = linear_interpolation(rnn_init1, rnn_1, lsmod, pnmod, info_layer, α, "rnn")
F_rnn2, _ = linear_interpolation(rnn_init2, rnn_2, lsmod, pnmod, info_layer, α, "rnn")
F_rnn3, _ = linear_interpolation(rnn_init3, rnn_3, lsmod, pnmod, info_layer, α, "rnn")
# LSTMs
printstyled("Starting LSTM Linear Interpolation...\n", color=:cyan)
F_lstm1, _ = linear_interpolation(lstm_init1, lstm_1, lsmod, pnmod, info_layer, α, "lstm")
F_lstm2, _ = linear_interpolation(lstm_init2, lstm_2, lsmod, pnmod, info_layer, α, "lstm")
F_lstm3, _ = linear_interpolation(lstm_init3, lstm_3, lsmod, pnmod, info_layer, α, "lstm")
# GRUs
printstyled("Starting GRU Linear Interpolation...\n", color=:cyan)
F_gru1, _ = linear_interpolation(gru_init1, gru_1, lsmod, pnmod, info_layer, α, "gru")
F_gru2, _ = linear_interpolation(gru_init2, gru_2, lsmod, pnmod, info_layer, α, "gru")
F_gru3, _ = linear_interpolation(gru_init3, gru_3, lsmod, pnmod, info_layer, α, "gru")

# Plotting
x = α

# RNN
y = hcat(F_rnn1, F_rnn2, F_rnn3)
p_rnn = plot(x, y, xlabel="\\alpha", ylabel="F", label="",title="RNN Linear Interpolation", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
		vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
#save("../../../Coadjoint_Method/submit/images/rnn_linear_interp.png", p_rnn)

# LSTM
y = hcat(F_lstm1, F_lstm2, F_lstm3)
p_lstm = plot(x, y, xlabel="\\alpha", ylabel="F", label="",title="LSTM Linear Interpolation", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
		vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
#save("../../../Coadjoint_Method/submit/images/lstm_linear_interp.png", p_lstm)

# GRU
y = hcat(F_gru1, F_gru2, F_gru3)
p_gru = plot(x, y, xlabel="\\alpha", ylabel="F", label="",title="GRU Linear Interpolation", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
		vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
#save("../../../Coadjoint_Method/submit/images/gru_linear_interp.png", p_gru)
