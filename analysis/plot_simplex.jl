# set directory to file location
cd("Desktop/NN_Implementation/training/opt_ntwks/")

include("../../src/neurons.jl")
include("../../src/network.jl")
include("../../src/partition.jl")
include("../../src/penalties.jl")
include("../../src/helpers.jl")

using JLD, Plots, SparseArrays, LinearAlgebra, CSV, DataFrames

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
			if sum(U[:,info_layer]) > 0
				label[1] = 1.0
			else
				label[2] = 1.0
			end
			U = collect(Iterators.flatten(U))
			U = vcat(U, zeros(ntwk_comb.hid_dim))
			X = Network.evaluate(ntwk_comb, U)
			Λ = Network.adjoint(ntwk_comb, X, label, loss)
			F[i] += loss.evaluate(label, X, ntwk_comb.results)
			G[i] += penalty.evaluate(Λ, ntwk_comb.features, ntwk_comb.hid_dim, ntwk_comb.seq_length)
		end
	end
	return F, G
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
adj_ntwk1 = load("toy_adjoint_8_39.jld", "ntwk") #32
adj_ntwk2 = load("toy_adjoint_8_32.jld", "ntwk")
adj_ntwk3 = load("toy_adjoint_8_23.jld", "ntwk")
co_ntwk1 = load("toy_coadjoint_8_39.jld", "ntwk")
co_ntwk2 = load("toy_coadjoint_8_32.jld", "ntwk")# 22, 24
co_ntwk3 = load("toy_coadjoint_8_23.jld", "ntwk")# 23, 25

#co_ntwk = load("toy_coadjoint_8_39.jld", "ntwk")
init_ntwk = load("toy_init_8_39.jld", "ntwk")
rand_ntwk = load("toy_random_pt.jld", "ntwk")

"""
Linear Interpolation Plots
"""
lsmod = loss.crossEntropy
pnmod = penalty.var_phi
info_layer = 6#8
mesh = 0:0.05:2
α = map(i -> mesh[i], 1:length(mesh))
F_adj1, _ = linear_interpolation(init_ntwk, adj_ntwk1, lsmod, pnmod, info_layer, α)
F_adj2, _ = linear_interpolation(init_ntwk, adj_ntwk2, lsmod, pnmod, info_layer, α)
F_adj3, _ = linear_interpolation(init_ntwk, adj_ntwk3, lsmod, pnmod, info_layer, α)
F_co1, _ = linear_interpolation(init_ntwk, co_ntwk1, lsmod, pnmod, info_layer, α)
F_co2, _ = linear_interpolation(init_ntwk, co_ntwk2, lsmod, pnmod, info_layer, α)
F_co3, _ = linear_interpolation(init_ntwk, co_ntwk3, lsmod, pnmod, info_layer, α)
F_coadj, _ = linear_interpolation(co_ntwk1, co_ntwk2, lsmod, pnmod, info_layer, α)



x = α
y = hcat(F, F_co) ./ 1000
plot(x,y)
y = hcat(F_adj1, F_adj2, F_adj3) ./ 1000
y2 = hcat(F_co1, F_co2, F_co3) ./ 1000
y3 = F_coadj ./ 1000
p=plot(x,y,xlabel="α", ylabel="J(θ)", label="",title="Adjoint Linear Interpolation", bg = RGB(0.2, 0.2, 0.2), legend=:bottomright)
vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
p2=plot(x,y2,xlabel="α", ylabel="J(θ)", label="",title="Coadjoint Linear Interpolation", bg = RGB(0.2, 0.2, 0.2),legend=:bottomright)
vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
p_coadj=plot(x,y3,xlabel="α", ylabel="J(θ)", label="",title="Coadjoint Linear Interpolation", bg = RGB(0.2, 0.2, 0.2),legend=:bottomright)
vline!([1.0],color=:red,lw=0.75,linestyle=:dash,label="")
#save("../../../Coadjoint_Method/soln_analysis/images/linear_interp.png", p)
"""
Generate Plotting Data
"""
f = plot_4pts(adj_ntwk, coadj_ntwk, init_ntwk, rand_ntwk, lsmod)

"""
Saving Results
"""
α = β = 0:0.05:1
using DataFrames, CSV
df = DataFrame(alpha=α, beta=β)
dfJ = DataFrame(f)


CSV.write("alpha_beta3.csv",  df, writeheader=true)
CSV.write("J3.csv",  dfJ, writeheader=false)


function twoSimplex(N :: Int64)
	κ = 2.0 / (N-1)
	mesh = 0:κ:2.0
	grid = Iterators.collect(Iterators.product(mesh, mesh))

	return grid
end

function F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, loss :: Module, simplex, info_layer :: Int64)
	ntwk = deepcopy(adj_ntwk)
	L = length(ntwk.neurons)
	F = zeros(size(simplex))
	N = 5000

	for i=1:size(simplex,1)
		for k=1:size(simplex,2)
			i_coor = 2.0 - simplex[i,k][1] - simplex[i,k][2]
			for j=1:L
				ntwk.neurons[j].β = (simplex[i,k][1]*adj_ntwk.neurons[j].β) + (simplex[i,k][2]*coadj_ntwk.neurons[j].β) + (i_coor*init_ntwk.neurons[j].β)
			end

			for j=1:N
				U, label = randn(1,10), zeros(ntwk.results)
				if sum(U[:,info_layer]) > 0
					label[1] = 1.0
				else
					label[2] = 1.0
				end
				U = collect(Iterators.flatten(U))
				U = vcat(U, zeros(ntwk.hid_dim))
				X = Network.evaluate(ntwk, U)
				Λ = Network.adjoint(ntwk, X, label, lsmod)
				F[i,k] += lsmod.evaluate(label, X, ntwk.results) / N
			end
		end
	end

	return F
end


s = twoSimplex(6)
lsmod = loss.crossEntropy
seeds = 1
function writeDataToFile(seeds)
	N = length(seeds)
	for i=1:N
		adj_ntwk = load("toy_exp/rnn_adjoint_2_$(seeds[i])_50.jld", "ntwk")
		coadj_ntwk = load("toy_exp/rnn_coadjoint_2_$(seeds[i])_50.jld", "ntwk")
		init_ntwk = load("toy_exp/rnn_init_2_$(seeds[i]).jld", "ntwk")
		F = F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, lsmod, s, 1)
		dfJ = DataFrame(F)
		CSV.write("layer_1/J_$(seeds[i]).csv",  dfJ, writeheader=false)
		if i == 1
			printstyled("\t\tProcess:\t...\t$(round(i/N*100,digits=2))%", color=:green)
		elseif i == N
			printstyled("\t\t100%.\n\t\t****** Completed ******", color=:green)
		else
			printstyled("...$(round(i/N*100,digits=2))%", color=:green)
		end
	end
end

writeDataToFile(seeds)
