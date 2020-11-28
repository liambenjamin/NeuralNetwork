

module Plot

using FileIO, IterTools, PyPlot, DataFrames, Main.Neurons, Main.Network, Main.Helpers#, JLD2,FileIO, Plots, SparseArrays, CSV, DataFrames

function unitCoord_interpolate(ntwk1 :: Network.network, ntwk2 :: Network.network, ntwk3 :: Network.network, ntwk4 :: Network.network, α :: Float64, β :: Float64)
	ntwk = deepcopy(ntwk1)
	L = length(ntwk.neurons)
	c = 1-α
	d = 1-β
	for i=1:L
		ntwk.neurons[i].β = (ntwk1.neurons[i].β*c*d)+(ntwk2.neurons[i].β*α*d)+(ntwk3.neurons[i].β*c*β)+(ntwk4.neurons[i].β*α*β)
	end
	return ntwk
end
function fourCoord_interpolate(ntwk1 :: Network.network, ntwk2 :: Network.network, ntwk3 :: Network.network, ntwk4 :: Network.network, α :: Float64, β :: Float64)
	ntwk = deepcopy(ntwk1)
	L = length(ntwk.neurons)
	x1, x2 = 0.0, 1.0
	y1, y2 = 0.0, 1.0
	K = 1 / ((x2-x1)*(y2-y1))
	a = x2 - α
	b = α - x1
	c = y2 - β
	d = β - y1
	for i=1:L
		Q1 = ntwk1.neurons[i].β
		Q2 = ntwk2.neurons[i].β
		Q3 = ntwk3.neurons[i].β
		Q4 = ntwk4.neurons[i].β
		term1 = c * (a*Q1 + b*Q3)
		term2 = d * (a*Q3 + b*Q4)
		ntwk.neurons[i].β = K * (term1 + term2)
	end
	return ntwk
end

function fourPt_interpolation(ntwk1 :: Network.network, ntwk2 :: Network.network, ntwk3 :: Network.network, ntwk4 :: Network.network, loss :: Module, penalty :: Module, ntwk_type :: String, dataset :: String, num_samples :: Int64)

	mesh = collect(-0.25:0.05:1.25)
	k = length(mesh)

	F = zeros(k,k)
	G = zeros(k,k)
	#Λ = size(ntwk1.graph[1])
	error = zeros(k,k)

	if dataset == "toy"
		x, y = Helpers.generate_dataset(num_samples, 20, 2)
	else
		feats, labels = Helpers.load_mnist_test()
		x, y = feats[1:num_samples], labels[1:num_samples,:]
	end

	for (α,β) in Iterators.product(1:k,1:k)
		#α = convert(Int64, α)
		#β = convert(Int64, β)

		if dataset == "toy"
			ntwk = unitCoord_interpolate(ntwk1, ntwk2, ntwk3, ntwk4, mesh[α], mesh[β])
			F[α,β], G[α,β], _, error[α,β] = Helpers.eval_binary_loss(ntwk, x, y, loss, penalty, ntwk_type)
		else
			ntwk = fourCoord_interpolate(ntwk1, ntwk2, ntwk3, ntwk4, mesh[α], mesh[β])
			F[α,β], error[α,β] = Helpers.mnist_test_error(ntwk, x, y, loss, penalty, ntwk_type)
		end
	end
	return F, error
end

function unitsq_interpolation(ntwk1 :: Network.network, ntwk2 :: Network.network, ntwk3 :: Network.network, ntwk4 :: Network.network, loss :: Module, penalty :: Module, ntwk_type :: String, dataset :: String, num_samples :: Int64)

	mesh = collect(0:0.05:1)
	k = length(mesh)

	F = zeros(k,k)
	G = zeros(k,k)
	#Λ = size(ntwk1.graph[1])
	error = zeros(k,k)

	if dataset == "toy"
		x, y = Helpers.generate_dataset(num_samples, 20, 2)
	else
		feats, labels = Heplers.load_mnist_test()
		x, y = feats[1:num_samples], labels[1:num_samples,:]
	end

	for (α,β) in Iterators.product(1:k,1:k)
		#α = convert(Int64, α)
		#β = convert(Int64, β)
		ntwk = unitCoord_interpolate(ntwk1, ntwk2, ntwk3, ntwk4, mesh[α], mesh[β])

		if dataset == "toy"
			F[α,β], G[α,β], _, error[α,β] = Helpers.eval_binary_loss(ntwk, x, y, loss, penalty, ntwk_type)
		else
			F[α,β], error[α,β] = Helpers.mnist_test_error(ntwk, x, y, loss, penalty, ntwk_type)
		end
	end
	return F, error
end

function recoverAdjoints(ntwk :: Network.network, loss :: Module, ntwk_type :: String)
	N = 500
	λ_ind = Helpers.ntwkX_ind(ntwk, ntwk_type)
	Λ = zeros(length(λ_ind),N)
	train_x, train_y = Helpers.generate_dataset(N, 20, 2)
	for i=1:N
		U, label = collect(Iterators.flatten(vcat(train_x[i,:,:], zeros(ntwk.hid_dim)))), train_y[i,:]
		X = Network.evaluate(ntwk, U)
		λ = Network.adjoint(ntwk, X, label, loss)
		Λ[:,i] = λ[λ_ind]
	end
	return Λ
end

"""
Description:
	Given an array of perturbantions (strings of the form "u1", "u28" etc),
	returns vector of unpacked perturbation statistics
Returns:
	1. Adjoint Pertubed Classification Error
	2. Coadjoint Perturbed Classification Error
	3. Adjoint Perturbed Test Error (F)
	4. Coadjoint Perturbed Test Error (F)

	- Note: First element in each of the 4 returned vectors are the un-perturbed
			classication and test errors.
"""
function unpackNtwkInfo(names :: Array, uIndx :: String, k :: Int64)

	adj_E = zeros(length(names))
    adj_ϵ_E = zeros(length(names))
    co_E = zeros(length(names))
    co_ϵ_E = zeros(length(names))

	adj_F = zeros(length(names))
    adj_ϵ_F = zeros(length(names))
    co_F = zeros(length(names))
    co_ϵ_F = zeros(length(names))

    for i=1:length(names)
		# load saved ntwk test information
		if k == 1
			adj_org = load("adjoint/mnist_rnn_adjoint_org_$(uIndx)_$(names[i]).jld2", "output")
	        adj_pert = load("adjoint/mnist_rnn_adjoint_pert_$(uIndx)_$(names[i]).jld2", "output")
	        co_org = load("adjoint/mnist_rnn_coadjoint_1e2_org_$(uIndx)_$(names[i]).jld2", "output")
	        co_pert = load("adjoint/mnist_rnn_coadjoint_1e2_pert_$(uIndx)_$(names[i]).jld2", "output")
		else
			adj_org = load("adjoint/mnist_rnn_adjoint_org_$(uIndx)_$(names[i])_d$(k).jld2", "output")
	        adj_pert = load("adjoint/mnist_rnn_adjoint_pert_$(uIndx)_$(names[i])_d$(k).jld2", "output")
	        co_org = load("adjoint/mnist_rnn_coadjoint_1e2_org_$(uIndx)_$(names[i])_d$(k).jld2", "output")
	        co_pert = load("adjoint/mnist_rnn_coadjoint_1e2_pert_$(uIndx)_$(names[i])_d$(k).jld2", "output")
		end
		# store classification error(s)
        adj_E[i] = adj_org[!,3][]
        adj_ϵ_E[i] = adj_pert[!,3][]
		co_E[i] = co_org[!,3][]
        co_ϵ_E[i] = co_pert[!,3][]
		# store test loss(es)
		adj_F[i] = adj_org[!,1][]
        adj_ϵ_F[i] = adj_pert[!,1][]
		co_F[i] = co_org[!,1][]
        co_ϵ_F[i] = co_pert[!,1][]
    end

	# form: ϵ = [0, smallest perturbation, ..., largest perturbation]
	adj_ϵ_error = vcat(adj_E[1], adj_ϵ_E)
	co_ϵ_error = vcat(co_E[1], co_ϵ_E)
	adj_ϵ_testF = vcat(adj_F[1], adj_ϵ_F)
	co_ϵ_testF = vcat(co_F[1], co_ϵ_F)

    return adj_ϵ_error, co_ϵ_error, adj_ϵ_testF, co_ϵ_testF
end
"""
Description:
	Given two networks, linearly interpolates the parameters of the networks
Returns:
	Interpolated test error (F) and test classification error (E)
"""
function linear_interpolation(ntwk_1 :: Network.network, ntwk_2 :: Network.network, loss :: Module, penalty :: Module, ntwk_type :: String)
	L = length(ntwk_1.neurons)
	N = 1000
	ntwk = deepcopy(ntwk_1)
	α = collect(0:0.025:1.0)
	F = zeros(length(α))
	E = zeros(length(α))
	test_x, test_y = Helpers.load_mnist_test()
	test_x, test_y = test_x[1:2500], test_y[1:2500,:]

	for i=1:length(α)
		# linear combination of ntwk1 and ntwk2
		for j=1:L
			ntwk.neurons[j].β = ((1 - α[i]) * ntwk_1.neurons[j].β) + (α[i] * ntwk_2.neurons[j].β)
		end
		# compute J(θ) for N samples
		F[i], E[i] = Helpers.mnist_test_error(ntwk, test_x, test_y, loss, ntwk_type)
		printstyled("Completed interpolation:\t[$(i)/$(length(α))]\r", color=:yellow)
	end
	return F, E
end

end # end plot module
