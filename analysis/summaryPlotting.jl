# set directory to file location
cd("Desktop/NN_Implementation/training/toy_exp/")

include("../../src/neurons.jl")
include("../../src/network.jl")
include("../../src/partition.jl")
include("../../src/penalties.jl")
include("../../src/helpers.jl")
include("../../src/rnn.jl")

using JLD2, Plots, SparseArrays, LinearAlgebra, CSV, DataFrames

"""
Linear Interpolation of Loss Surface
θ = (1-α) × θι + α × θɾ
"""

function linear_interpolation(ntwk_1, ntwk_2, loss :: Module, penalty :: Module, info_layer :: Int64, alpha)
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
			U = vcat(U, zeros(ntwk_comb.hid_dim))
			X = Network.evaluate(ntwk_comb, U)
			Λ = Network.adjoint(ntwk_comb, X, label, loss)
			F[i] += loss.evaluate(label, X, ntwk_comb.results)
			G[i] += penalty.evaluate(Λ, ntwk_comb.features, ntwk_comb.hid_dim, ntwk_comb.seq_length, "rnn")
		end
	end
	return F ./ N, G ./ N
end

"""
Simplex between 4 points
"""
function plot_4pts(ntwk, co_ntwk, init_ntwk, random_ntwk, loss :: Module)
	α = β = 0:0.05:1
	N = 1000
	ntwk_comb = deepcopy(ntwk)
	L = length(ntwk.neurons)
	F = zeros(length(α), length(β))
	for i=1:length(α)
		for k=1:length(β)
			for j=1:L
				d1 = ntwk.neurons[j].β - init_ntwk.neurons[j].β
				d2 = co_ntwk.neurons[j].β - init_ntwk.neurons[j].β
				ϕ = α[i] * d1 + init_ntwk.neurons[j].β
				ψ = α[i] * d2 + init_ntwk.neurons[j].β
				θ = β[k] * ϕ + (1-β[k]) * ψ
				ntwk_comb.neurons[j].β = θ
			end

			# find loss
			for j=1:N
				U, label = randn(1,10), zeros(ntwk_comb.results)
				if sum(U[:,info_layer]) > 0
					label[1] = 1.0
				else
					label[2] = 1.0
				end
				U = collect(Iterators.flatten(U))
				U = vcat(U, zeros(ntwk_comb.hid_dim))
				X = Network.evaluate(ntwk_comb, U)
				Λ = Network.adjoint(ntwk_comb, X, label, lsmod)
				F[i,k] += lsmod.evaluate(label, X, ntwk_comb.results) / N
			end
		end
		println("$(floor(i/length(α)*100)) % finished...")
	end
	return F
end

"""
Loading Trained Networks
"""
#cd("rnn_epochs")
adj_ntwk1 = load("rnn_adjoint_4_1_50.jld", "ntwk")
co_ntwk1 = load("rnn_coadjoint_4_1_50.jld", "ntwk")
init_ntwk1 = load("rnn_init_4_1.jld", "ntwk")

"""
Linear Interpolation Plots
"""
lsmod = loss.crossEntropy
pnmod = penalty.var_phi
info_layer = 4
mesh = 0:0.05:2
α = map(i -> mesh[i], 1:length(mesh))
F_adj, _ = linear_interpolation(init_ntwk1, adj_ntwk1, lsmod, pnmod, info_layer, α)
F_co, _ = linear_interpolation(init_ntwk1, co_ntwk1, lsmod, pnmod, info_layer, α)
F_adj_co = linear_interpolation(adj_ntwk1, co_ntwk1, lsmod, pnmod, info_layer, α)


x = α
y = F_adj
y2 = F_co
y3 = F_adj_co
p_adj = plot(x,y,xlabel="α", ylabel="J(θ)", label="",title="Adjoint Linear Interpolation", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
		vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
p_co = plot(x,y2,xlabel="α", ylabel="J(θ)", label="",title="Coadjoint Linear Interpolation", bg = RGB(0.2, 0.2, 0.2),legend=:bottomright)
		vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
p_adj_co = plot(x,y3,xlabel="α", ylabel="J(θ)", label="",title="Coadjoint Linear Interpolation", bg = RGB(0.2, 0.2, 0.2),legend=:bottomright)
		vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
#save("../../../Coadjoint_Method/soln_analysis/images/linear_interp.png", p)


"""
Plots the influence of the magnitude of G on optimization path
"""

function G_influence(info_layer, seed)

	adj_E, adj_F = zeros(50), zeros(50)
	co1_E, co1_F = zeros(50), zeros(50)
	co2_E, co2_F = zeros(50), zeros(50)
	co3_E, co3_F = zeros(50), zeros(50)

	for i=1:50 #number of checkpoints saved
		adj_ntwk = load("rnn_epochs/G1e-0/rnn_adjoint_$(info_layer)_$(seed)_$(i).jld", "ntwk")
		co_ntwk1 = load("rnn_epochs/G1e-0/rnn_coadjoint_$(info_layer)_$(seed)_$(i).jld", "ntwk")
		co_ntwk2 = load("rnn_epochs/G1e-1/rnn_coadjoint_$(info_layer)_$(seed)_$(i).jld", "ntwk")
		co_ntwk3 = load("rnn_epochs/G1e-2/rnn_coadjoint_$(info_layer)_$(seed)_$(i).jld", "ntwk")


		adj_E[i], adj_F[i] = RNN.test_error(adj_ntwk, info_layer, lsmod)
		co1_E[i], co1_F[i] = RNN.test_error(co_ntwk1, info_layer, lsmod)
		co2_E[i], co2_F[i] = RNN.test_error(co_ntwk2, info_layer, lsmod)
		co3_E[i], co3_F[i] = RNN.test_error(co_ntwk3, info_layer, lsmod)
	end
	return adj_E, adj_F, co1_E, co1_F, co2_E, co2_F, co3_E, co3_F
end


adj_E, adj_F, co1_E, co1_F, co2_E, co2_F, co3_E, co3_F = G_influence(4, 1) # input: info_layer, seed
adj_E2, adj_F2, co1_E2, co1_F2, co2_E2, co2_F2, co3_E2, co3_F2 = G_influence(4, 2)

x = 1:length(adj_E)
F = hcat(adj_F, co1_F, co2_F, co3_F)
Y = hcat(adj_E, co1_E, co2_E, co3_E)
F2 = hcat(adj_F2, co1_F2, co2_F2, co3_F2)
Y2 = hcat(adj_E2, co1_E2, co2_E2, co3_E2)

p1 = plot(x,F,xlabel="Epoch", ylabel="F", label=["F" "F + G" "F + 0.1 G" "F + 0.01 G"],title="Influence of G on Minimization of F", bg = RGB(0.2, 0.2, 0.2), legend=:right)
p2 = plot(x,F2,xlabel="Epoch", ylabel="F", label=["F" "F + G" "F + 0.1 G" "F + 0.01 G"],title="Influence of G on Minimization of F", bg = RGB(0.2, 0.2, 0.2), legend=:right)
p3 = plot(x,Y,xlabel="Epoch", ylabel="F", label=["F" "F + G" "F + 0.1 G" "F + 0.01 G"],title="Influence of G on Classification Error", bg = RGB(0.2, 0.2, 0.2), legend=:right)
p4 = plot(x,Y2,xlabel="Epoch", ylabel="F", label=["F" "F + G" "F + 0.1 G" "F + 0.01 G"],title="Influence of G on Classification Error", bg = RGB(0.2, 0.2, 0.2), legend=:right)

save("../../../Coadjoint_Method/soln_analysis/images/g_influence_F1.png", p1)
save("../../../Coadjoint_Method/soln_analysis/images/g_influence_F2.png", p3)

"""
Generating metadata for Matlab Surface Interpolation Plots
"""
α = β = 0:2.0/20:2.0

df = DataFrame(alpha=α, beta=β)


CSV.write("alpha_beta_21.csv",  df, writeheader=true)


function twoSimplex(N :: Int64)
	κ = 2.0 / (N-1)
	mesh = 0:κ:2.0
	grid = Iterators.collect(Iterators.product(mesh, mesh))

	return grid
end

function F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, loss :: Module, simplex, info_layer :: Int64)
	ntwk = deepcopy(adj_ntwk)
	L = length(ntwk.neurons)
	E = zeros(size(simplex)) # classification error over test set
	F = zeros(size(simplex)) # training error (F) over test set
	for i=1:size(simplex,1)
		for k=1:size(simplex,2)
			i_coor = 1.0 - sum(simplex[i,k])
			for j=1:L
				ntwk.neurons[j].β = (simplex[i,k][1]*adj_ntwk.neurons[j].β) + (simplex[i,k][2]*coadj_ntwk.neurons[j].β) + (i_coor*init_ntwk.neurons[j].β)
			end
			E[i,k], F[i,k] = RNN.test_error(ntwk, info_layer, loss) # 4000 samples

		end
	end
	return F, E
end


s = twoSimplex(21)
lsmod = loss.crossEntropy
seeds = 1:10
info_layer = 4
function writeDataToFile(seeds, info_layer :: Int64)
	N = length(seeds)
	for i=1:N
		adj_ntwk = load("rnn_epochs/G1e-0/rnn_adjoint_$(info_layer)_$(seeds[i])_50.jld", "ntwk")
		coadj_ntwk = load("rnn_epochs/G1e-2/rnn_coadjoint_$(info_layer)_$(seeds[i])_50.jld", "ntwk")
		init_ntwk = load("rnn_epochs/rnn_init_$(info_layer)_$(seeds[i]).jld", "ntwk")
		F, E = F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, lsmod, s, info_layer)
		dfJ = DataFrame(F)
		CSV.write("simplexF21_$(info_layer)_$(seeds[i]).csv",  dfJ, writeheader=false)
		dfJE = DataFrame(E)
		CSV.write("simplexErr21_$(info_layer)_$(seeds[i]).csv",  dfJE, writeheader=false)
		if i == 1
			printstyled("\t\tProcess:\t...\t$(round(i/N*100,digits=2))%", color=:green)
		elseif i == N
			printstyled("\t\t100%.\n\t\t****** Completed ******", color=:green)
		else
			printstyled("...$(round(i/N*100,digits=2))%", color=:green)
		end
	end
end

writeDataToFile(seeds, info_layer)
