include("../../src/neurons.jl")
include("../../src/network.jl")
include("../../src/partition.jl")
include("../../src/penalties.jl")
include("../../src/helpers.jl")
include("../../src/rnn.jl")
include("../../src/lstm.jl")
include("../../src/gru.jl")

using FileIO, JLD2, Plots, SparseArrays, LinearAlgebra, CSV, DataFrames

"""
Linear Interpolation of Loss Surface
θ = (1-α) × θι + α × θɾ
"""

function adjointSize(loss :: Module, penalty :: Module, ntwk_type :: String)
	N = 1000 # test samples
	F = zeros(200) # number of epoch
	G = zeros(200)
	λ_2 = zeros(200)
	λ_5 = zeros(200)
	λ_8 = zeros(200)

	# compute J(θ) for N samples
	for i=1:200 #number of epochs
		ntwk = load("$(ntwk_type)/$(ntwk_type)_adjoint_1_$(i).jld2", "ntwk")
		for k=1:N
			U, label = randn(1,10), zeros(ntwk.results)
			if sum(U) > 0
				label[1] = 1.0
			else
				label[2] = 1.0
			end
			U = collect(Iterators.flatten(U))
			U = ntwk_type != "lstm" ?
				vcat(U, zeros(ntwk.hid_dim)) :
				vcat(U, zeros(2*ntwk.hid_dim))
			X = Network.evaluate(ntwk, U)
			Λ = Network.adjoint(ntwk, X, label, loss)
			F[i] += loss.evaluate(label, X, ntwk.results)
			G[i] += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type)

			if ntwk_type == "rnn"
				adjoint_ind = RNN.rnnXind(ntwk.features-ntwk.hid_dim, ntwk.hid_dim, 10)
			elseif ntwk_type == "lstm"
				adjoint_ind = LSTM.lstmXind(ntwk.features-2*ntwk.hid_dim, ntwk.hid_dim, 10)
			else
				adjoint_ind = GRU.gruXind(ntwk.features-ntwk.hid_dim, ntwk.hid_dim, 10)
			end

			λ_2[i] += norm(Λ[adjoint_ind[ntwk.hid_dim*1+1:2*ntwk.hid_dim]])
			λ_5[i] += norm(Λ[adjoint_ind[ntwk.hid_dim*4+1:5*ntwk.hid_dim]])
			λ_8[i] += norm(Λ[adjoint_ind[ntwk.hid_dim*7+1:8*ntwk.hid_dim]])
		end
	end

	return F ./ N, G ./ N, λ_2 ./ N, λ_5 ./ N, λ_8 ./ N
end

"""
Load networks
"""
lsmod = loss.crossEntropy
pnmod = penalty.var_phi

F_rnn, G_rnn, λ2_rnn, λ5_rnn, λ8_rnn = adjointSize(lsmod, pnmod, "rnn")
F_lstm, G_lstm, λ2_lstm, λ5_lstm, λ8_lstm = adjointSize(lsmod, pnmod, "lstm")
F_gru, G_gru, λ2_gru, λ5_gru, λ8_gru = adjointSize(lsmod, pnmod, "gru")


# Plotting
x = 1:200

# RNN
y = hcat(λ2_rnn, λ5_rnn, λ8_rnn)
p_rnn = plot(x, y, xlabel="Training Batch", ylabel="||\\lambda||", label=["||\\lambda_{2}||" "||\\lambda_{5}||" "||\\lambda_{8}||"], title="RNN Adjoint Size", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
save("../../../Coadjoint_Method/submit/images/rnn_adjoint_size.png", p_rnn)
# LSTM
y = hcat(λ2_lstm, λ5_lstm, λ8_lstm)
p_lstm = plot(x, y, xlabel="Training Batch", ylabel="||\\lambda||", label=["||\\lambda_{2}||" "||\\lambda_{5}||" "||\\lambda_{8}||"],title="LSTM Adjoint Size", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
save("../../../Coadjoint_Method/submit/images/lstm_adjoint_size.png", p_lstm)
# GRU
y = hcat(λ2_gru, λ5_gru, λ8_gru)
p_gru = plot(x, y, xlabel="Training Batch", ylabel="||\\lambda||", label=["||\\lambda_{2}||" "||\\lambda_{5}||" "||\\lambda_{8}||"],title="GRU Adjoint Sizes", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
save("../../../Coadjoint_Method/submit/images/gru_adjoint_size.png", p_gru)
# RNN, LSTM and GRU
# GRU
y = hcat(λ2_rnn, λ2_lstm, λ2_gru)
p_mix = plot(x, y, xlabel="Training Batch", ylabel="||\\lambda||", label=["RNN" "LSTM" "GRU"],title="Adjoint Sizes", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
save("../../../Coadjoint_Method/submit/images/adjoint_size.png", p_mix)
