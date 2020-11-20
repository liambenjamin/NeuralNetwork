# set directory to file location
#cd("Desktop/NN_Implementation/training//")

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")

using JLD2, PyPlot, SparseArrays, LinearAlgebra, CSV, DataFrames, FileIO, MLDatasets

#=
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
	κ = 1.25 / (N-1)
	mesh = 0:κ:1.25
	grid = Iterators.collect(Iterators.product(mesh, mesh))

	return grid
end

function F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, loss :: Module, simplex)

	# load MNIST data
	train_x, train_y = MNIST.traindata()
	test_x,  test_y  = MNIST.testdata()
	# reshape features and one-hot encode labels
	train_x = map(i -> collect(Iterators.flatten(train_x[:,:,i])), 1:size(train_x,3))
	test_x = map(i -> collect(Iterators.flatten(test_x[:,:,i])), 1:size(test_x,3))
	train_y, test_y = Helpers.one_hot(train_y), Helpers.one_hot(test_y)

	ntwk = deepcopy(adj_ntwk)
	L = length(ntwk.neurons)
	#F = zeros(size(simplex))
	F_test = zeros(size(simplex))
	F_train = zeros(size(simplex))
	test_error = zeros(size(simplex))
	train_error = zeros(size(simplex))

	#N = 500

	for i=1:size(simplex,1)
		for k=1:size(simplex,2)
			i_coor = 1.25 - simplex[i,k][1] - simplex[i,k][2]
			if i_coor <= 1.25
				for j=1:L
					ntwk.neurons[j].β = (simplex[i,k][1]*adj_ntwk.neurons[j].β) + (simplex[i,k][2]*coadj_ntwk.neurons[j].β) + (i_coor*init_ntwk.neurons[j].β)
				end

				F_test[i,k], test_error[i,k] = Helpers.mnist_test_error(ntwk, test_x[1:300], test_y[1:300,:], loss, "rnn")
				F_train[i,k], train_error[i,k] = Helpers.mnist_test_error(ntwk, train_x[1:300], train_y[1:300,:], loss, "rnn")
			end
		end
	end

	return F_test, F_train, test_error, train_error
end



dir = "mnist"

#s = twoSimplex(21)

lsmod = loss.softmaxCrossEntropy
seeds = [1,2]

function writeDataToFile(seeds, loss :: Module, simplex)
	N = length(seeds)
	for i=1:N
		adj_ntwk = load("$(dir)/rnn_adjoint_5.jld2", "ntwk")
		coadj_ntwk = load("$(dir)/rnn_coadjoint_1e$(seeds[i])_5.jld2", "ntwk")
		init_ntwk = load("$(dir)/rnn_random_init.jld2", "ntwk")
		F_test, F_train, test_error, train_error = F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, lsmod, simplex)
		dfF_test = DataFrame(F_test)
		dfF_train = DataFrame(F_train)
		dfE_test = DataFrame(test_error)
		dfE_train = DataFrame(train_error)
		CSV.write("$(dir)/JF_test_1e$(i).csv",  dfF_test, writeheader=false)
		CSV.write("$(dir)/JF_train_1e$(i).csv",  dfF_train, writeheader=false)
		CSV.write("$(dir)/JE_test_1e$(i).csv",  dfE_test, writeheader=false)
		CSV.write("$(dir)/JE_train_1e$(i).csv",  dfE_train, writeheader=false)
		if i == 1
			printstyled("\t\tProcess:\t...\t$(round(i/N*100,digits=2))%", color=:green)
		elseif i == N
			printstyled("\t\t100%.\n\t\t****** Completed ******", color=:green)
		else
			printstyled("...$(round(i/N*100,digits=2))%", color=:green)
		end
	end
end

writeDataToFile(seeds, lsmod, s)
=#


"""
Bilinear Interpolation between three points
 - Max = 1.00
"""

function simplex_3pt(N :: Integer)
	u = collect(0:(1/N):1.0)
	v = reverse(u)
	C = collect(Iterators.product(u,u))
	for i in CartesianIndices(C)
		if sum(C[i]) > 1.0
			C[i] = (NaN, NaN)
		end
	end
	return C
end


function F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, loss :: Module, simplex)

	# load MNIST data
	train_x, train_y = MNIST.traindata()
	test_x,  test_y  = MNIST.testdata()
	# reshape features and one-hot encode labels
	train_x = map(i -> collect(Iterators.flatten(train_x[:,:,i])), 1:size(train_x,3))
	test_x = map(i -> collect(Iterators.flatten(test_x[:,:,i])), 1:size(test_x,3))
	train_y, test_y = Helpers.one_hot(train_y), Helpers.one_hot(test_y)

	ntwk = deepcopy(adj_ntwk)
	L = length(ntwk.neurons)
	#F = zeros(size(simplex))
	F_test = zeros(size(simplex))
	F_train = zeros(size(simplex))
	test_error = zeros(size(simplex))
	train_error = zeros(size(simplex))

	for i=1:size(simplex,1)
		for k=1:size(simplex,2)

			u, v = simplex[i,k]
			if sum([u,v]) != NaN
				for j=1:L
					ntwk.neurons[j].β = u*adj_ntwk.neurons[j].β + v*coadj_ntwk.neurons[j].β + (1.0-u-v)*init_ntwk.neurons[j].β
				end

				F_test[i,k], test_error[i,k] = Helpers.mnist_test_error(ntwk, test_x[1:300], test_y[1:300,:], loss, "rnn")
				F_train[i,k], train_error[i,k] = Helpers.mnist_test_error(ntwk, train_x[1:300], train_y[1:300,:], loss, "rnn")
			end
		end
	end

	return F_test, F_train, test_error, train_error
end



C = simplex_3pt(20)
dir = "mnist"
lsmod = loss.softmaxCrossEntropy
adj_ntwk = load("$(dir)/rnn_adjoint_5.jld2", "ntwk")
coadj_ntwk = load("$(dir)/rnn_coadjoint_1e1_5.jld2", "ntwk")
init_ntwk = load("$(dir)/rnn_random_init.jld2", "ntwk")
F_test, F_train, test_error, train_error = F_simplex(adj_ntwk, coadj_ntwk, init_ntwk, lsmod, C)

CSV.write("$(dir)/F_test_1e1.csv",  DataFrame(F_test), writeheader=false)
CSV.write("$(dir)/F_train_1e1.csv",  DataFrame(F_train), writeheader=false)
CSV.write("$(dir)/E_test_1e1.csv",  DataFrame(test_error), writeheader=false)
CSV.write("$(dir)/E_train_1e1.csv",  DataFrame(train_error), writeheader=false)


F_test_flat = collect(Iterators.flatten(F_test))
x = []
y = []
z = []
for i=1:length(C)
	if isnan(sum(C[i]))
		nothing
	else
		append!(x, C[i][1])
		append!(y, C[i][2])
		append!(z, F_test_flat[i])
		#x[i] = C[i][1]
		#y[i] = C[i][2]
		#z[i] = F_test_flat[i]
	end
end

#surface(x,y,z, camera=(-30,30))

index = [1,6,21] # init, adj, coad
df = hcat(x,y,z)
lab_x = df[index,1]
lab_y = df[index,2]
lab_z = df[index,3]

plot(x,y,z, title="Linear Interpolation")

surface(x,y,z, title="Linear Interpolation", camera=(60,30))


scatter(lab_x,lab_y,lab_z, label="")
annotate!(lab_x[1],lab_y[1],lab_z[1], "init")
