seed = 1 #parse(Int64, ENV["seed"])

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("helpers.jl")


using LinearAlgebra, SparseArrays, MLDatasets, Random, Colors, Main.Neurons, Main.Network, Main.Helpers

# F objective and gradients
f(X, Y) = -Y' * log.(X[end-length(Y)+1:end]) # equivalent to label[pos_arg_max] * pred[pos_arg_max]

function df(X,Y)
    outInd = length(X)-length(Y)+1:length(X)
    ∂f = zeros(length(X))
    ∂f[outInd] = -Y .* X[outInd].^-1
    return ∂f
end

function d2f(X,Y)
    outInd = length(X)-length(Y)+1:length(X)
    ∂f2 = zeros(length(X), length(X))
    ∂f2[outInd, outInd] = diagm(0 => Y .* X[outInd].^-2)
    return ∂f2
end

# G objective and gradient (Var + Log Penalty)

function g(L)
    layer_T = 785:1364
    Λ = reshape(L[layer_T], 20, 29)
    d, L = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:L)
    M = sum(nrms)/L
    λ_star = 1e-5
    ϕ = map( i -> norm(Λ[:,L][i]) < norm(λ_star) ?
        # ϕ1
        -norm(λ_star) / (norm(Λ[:,L][i])*(1+log(norm(λ_star)^2))) + 1 :
        # ϕ2
        log(norm(λ_star)*norm(Λ[:,L][i])) / (1+log(norm(λ_star)^2)), 1:d)
    return (sum( nrms.^2)/L - M^2) + 25 * sum(ϕ)
end

function dg(L)
    layer_T = 785:1364
    Λ = reshape(L[layer_T], 20, 29)
    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    val = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : ((2/T)*Λ[:,i] - (2*M/T)* Λ[:,i]/nrms[i]) , 1:T)...)
    ∂ϕ = nrms[T] == 0 ? zeros(d) : map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ'1
        (norm(λ_star)*Λ[:,T][i]) / (norm(Λ[:,T][i])^3 * (1+log(norm(λ_star)^2))) :
        # ϕ'2
        (norm(λ_star)*Λ[:,T][i]) / (norm(λ_star) * (1+log(norm(λ_star)^2)) * norm(Λ[:,T][i])^2), 1:d)
    val[:,T] = nrms[T] == 0 ? zeros(d) : val[:,T] + 25 * ∂ϕ
    # append vector to adjoint vector
    grad = zeros(length(L))
    grad[layer_T] = collect(Iterators.flatten(val))
    return grad
end

function g_log(L)
    layer_T = 785:1364
    Λ = reshape(L[layer_T], 20, 29)
    T = size(Λ,2)
    log_nrms = map(i -> log(norm(Λ[:,i])^2), 1:T)
    return (-1/2) * sum(log_nrms)
end

function dg_log(L)
    layer_T = 785:1364
    Λ = reshape(L[layer_T], 20, 29)
    d,T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    dg_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : (-Λ[:,i]) ./ (norm(Λ[:,i])^2) , 1:T)...)
    dg = zeros(length(L))
    dg[layer_T] = collect(Iterators.flatten(dg_mat))
    return dg
end


"""
Implements MNIST RNN Training
"""

function train(ntwk, num_epochs :: Int64, f :: Function, df :: Function, d2f :: Function, g :: Function, dg :: Function, η :: Float64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}})

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs))

    # load and store
	train_x, train_y = MNIST.traindata()
	test_x,  test_y  = MNIST.testdata()
	# one hot encoding of labels
	train_y = Helpers.one_hot(train_y) .+ ones(size(train_y,1)) * 0.001
	test_y = Helpers.one_hot(test_y)

	# store sgd update time
	update_time = 0.0

	for i=1:num_epochs

		# Epoch Loss
		F = 0.0
		G = 0.0
		Λ = zeros(1414)

		# set epoch learning rate as function of initial learning rate
		l_rate = η #lr_scheduler(η, i)

		# shuffle training data
		perm_i = shuffle(1:size(train_x,3))
		train_x = train_x[:,:,perm_i]
		train_y = train_y[perm_i,:]

		for j=1:5#size(train_x, 3)
			U, label = collect(Iterators.flatten(train_x[:,:,j])), train_y[j,:]
			U = vcat(U, zeros(length(ntwk.neurons[1].β) - 28 - 1))
			X = Network.evaluate(ntwk, U)
			Λ = Network.adjoint(ntwk, X, label, df :: Function) # df = loss gradient

			F += f(X, label)
			G += g(Λ)

			# update ntwk params
			if method == "adjoint" #|| i > 3 #uncomment for transitioning from F+G --> F update
				update_time = @elapsed begin
					#update!(df, ntwk, U, label, l_rate) # no partition
					update!(df, ntwk, U, label, l_rate, syncFlag, partition) # with partition
				end
			else
				update_time = @elapsed begin
					#update!(df, d2f, dg, ntwk, U, label, l_rate) # no partition
					update!(df, d2f, dg, ntwk, U, label, l_rate, syncFlag, partition) # with partition
				end
			end

		end
		# classification error over test set
		error = Helpers.classification_error(ntwk, test_x, test_y)

		epoch_info["F"][i] = F
		epoch_info["G"][i] = G
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = update_time
		epoch_info["Λ_init"][i] = norm(Λ[785:804] ./ size(train_x,3))
		epoch_info["Λ_out"][i] = norm(Λ[1345:1364] ./ size(train_x,3))
		epoch_info["Error"][i] = error


		#printstyled("\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
		#printstyled("\tError: $(error)\n", color=:light_magenta)
		#printstyled("\tObjective (F+G): $(F+G)\n", color=:light_magenta)
		#printstyled("\tF: $(F)\n", color=:light_magenta)
		#printstyled("\tG: $(G)\n", color=:light_magenta)
		#printstyled("\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
		#printstyled("\t||λΤ||: $(epoch_info["Λ_out"][i])\n\n", color=:light_magenta)

		# WRITE EPOCH INFO TO FILE
		#epoch_file = "training_info/rnn_$(method)/sig_$(seed).dat"
	    #f = open(epoch_file, "a")
			printstyled("\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
			printstyled("\tTime: $(update_time)\n", color=:light_magenta)
			printstyled("\tError: $(error)\n", color=:light_magenta)
			printstyled("\tObjective (F+G): $(F+G)\n", color=:light_magenta)
			printstyled("\tF: $(F)\n", color=:light_magenta)
			printstyled("\tG: $(G)\n", color=:light_magenta)
			printstyled("\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
			printstyled("\t||λΤ||: $(epoch_info["Λ_out"][i])\n\n", color=:light_magenta)
		#close(f)

		# stop training if NaNs produced
		if isnan(F)
			return [epoch_info]
		end
	end
	return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
method = "coadjoint"
transition = "After epoch 3"
hid_state = 20
output_dim = 10
input_dim = 784 + hid_state
seq_length = 28
fc_dim = 30
neurons, rowInd, colInd = Helpers.specifyGraph(input_dim, hid_state, output_dim, seq_length, fc_dim)
vals = ones(Int64, length(rowInd))
graph = sparse(rowInd,colInd,vals)
ntwk = Network.network(neurons, input_dim, output_dim) # (neuron list, input dim, output dim)
Network.graph!(ntwk,graph)
# partition neurons to share weights
partition = Helpers.getPartition(length(ntwk.neurons), hid_state, output_dim, seq_length, fc_dim)
Partition.synchronizeParameters!(ntwk,partition)
# verify partition
syncFlag = Partition.verifyPartition(ntwk,partition)
# learning rate
η = 1e-4
num_epochs = 300
# write network training parameters to file
#epoch_file = "training_info/rnn_$(method)/sig_$(seed).dat"


#file = open(epoch_file, "a")
	printstyled("**********\tTraining Method: RNN $(method)\t**********\n\n", color = :yellow, bold = :true)
    printstyled("\tSeed: $(seed)\n", color = :light_cyan)
    printstyled("\tLearning Rate: $(η)\n", color = :light_cyan)
    printstyled("\tLoss (F): Cross Entropy (Softmax)\n", color = :light_cyan)
	printstyled("\tPenalty (G): Log (All Layers)\n", color = :light_cyan)
    #printstyled("\tPenalty (G): Σ var(|λι|) + 25×ϕ(λτ)\n", color = :light_cyan)
	printstyled("\tTransition (F+G → F): $(transition)\n\n", color = :light_cyan)
	printstyled("\tNetwork Architecture:\n", color=:light_cyan)
	printstyled("\t\tInput Dim: $(input_dim) (784 input + $(hid_state) init state)\n", color = :light_cyan)
	printstyled("\t\tHidden Dim: $(hid_state)\n", color = :light_cyan)
	printstyled("\t\tFully Connect Dim: $(fc_dim)\n", color = :light_cyan)
	printstyled("\t\tSoftmax Dim: $(output_dim)\n\n", color = :light_cyan)
	printstyled("**********************************************************\n\n", color = :yellow, bold = :true)
#close(file)

training_output = train(ntwk, num_epochs, f, df, d2f, g_log, dg_log, η, method, syncFlag, partition)

# save training output
#save("training_output/rnn_$(method)/$(method)_$(seed).jld", "data", training_output)
