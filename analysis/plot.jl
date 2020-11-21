

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

function unitsq_interpolation(ntwk1 :: Network.network, ntwk2 :: Network.network, ntwk3 :: Network.network, ntwk4 :: Network.network, loss :: Module, penalty :: Module, ntwk_type :: String)

	mesh = collect(0:0.05:1)
	k = length(mesh)

	F = zeros(k,k)
	G = zeros(k,k)
	#Λ = size(ntwk1.graph[1])
	error = zeros(k,k)

	x, y = Helpers.generate_dataset(1000,20,2)

	for (α,β) in Iterators.product(1:k,1:k)
		#α = convert(Int64, α)
		#β = convert(Int64, β)
		ntwk = unitCoord_interpolate(ntwk1, ntwk2, ntwk3, ntwk4, mesh[α], mesh[β])

		F[α,β], G[α,β], _, error[α,β] = Helpers.eval_binary_loss(ntwk, x, y, loss, penalty, ntwk_type)
	end
	return F, G, error
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

function unpackNtwkInfo(names :: Array, uIndx :: String)
    #F = zeros(length(names))
    #G = zeros(length(names))
    E = zeros(length(names))
    #Fp = zeros(length(names))
    #Gp = zeros(length(names))
    Ep = zeros(length(names))

    Ea = zeros(length(names))
    Eap = zeros(length(names))

    for i=1:length(names)
        f_org = load("adjoint/mnist_rnn_coadjoint_1e2_org_$(uIndx)_$(names[i])_.jld2", "output")
        f_pert = load("adjoint/mnist_rnn_coadjoint_1e2_pert_$(uIndx)_$(names[i])_.jld2", "output")

        adj_org = load("adjoint/mnist_rnn_org_$(uIndx)_$(names[i])_.jld2", "output")
        adj_pert = load("adjoint/mnist_rnn_pert_$(uIndx)_$(names[i])_.jld2", "output")
        E[i] = f_org[!,3][]
        Ep[i] = f_pert[!,3][]
        Ea[i] = adj_org[!,3][]
        Eap[i] = adj_pert[!,3][]
    end
    return reverse(E), reverse(Ep), reverse(Ea), reverse(Eap)
end


end # end plot module
