#Author: Liam Johnston
#Date: June 3, 2020
#Description: Functions used during training of neural networks


module Helpers

using FileIO, Random, LinearAlgebra, MultivariateStats, NPZ, MLDatasets, Statistics, SparseArrays, Main.Neurons, Main.Network, Main.Partition, Main.RNN, Main.LSTM, Main.GRU

#=

"""
Description:
    Embeds a single sample (review) of the integer valued data (of the IMDB Movie) into
    R^100 using an embedding matrix produced using the GloVe embedding algorithm.

Dependencies:
    - NPZ
Data Files:
	- Embedding Matrix (E)
		- E: ``npz(imdb-params/keras-rnn-imdb-embedding-matrix.npz)``
	- datasets:
	 	- training data: ``npz(imdb-params/keras-rnn-imdb-training-features.npz)``
		- testing data: ``npz(imdb-params/keras-rnn-imdb-testing-features.npz)``
"""
function embed(x :: Array, E :: Matrix)
    x_imbed = zeros(size(x,1), 100)
    #x_star = map(i -> E[i+1,:], x)
    for i=1:size(x,1)
        x_imbed[i,:] = E[x[i]+1,:]
    end
    return x_imbed
end

"""
Description:
	Returns test error for IMDB Sentiment Classification
Dependencies:
	LinearAlgebra
Input:
	1. network
	2. features: (25000,500)
	3. labels: (25000,)
	4. E: embedding matrix
	5. ntwk_type: string of either: `rnn`, `lstm` or `gru`
"""
function imdb_test_error(network :: Network.network, features, labels, E :: Matrix, ntwk_type :: String)
	#Verify prediction and true label are same size
	network.results != 1 && @error "Prediction and true label are different dimensions."

	N = size(features,1)
	num_correct = 0
	x0 = ntwk_type == "lstm" ? 2*network.hid_dim : network.hid_dim

	for i=1:N
		U, label = Helpers.embed(features[i,:],E), labels[i]
		U = vcat(collect(Iterators.flatten(U')), zeros(network.hid_dim))
		X = Network.evaluate(network, U)[end-network.results+1:end]
		X[end] > 0.5 ? num_correct += 1 : nothing
   end
   return 1.0 - num_correct/N
end

"""
Description:
	Returns test error for Reuters newswire classification task
Dependencies:
	LinearAlgebra
Input:
	1. network
	2. features: (# samples,150)
	3. labels: (# samples, 46)
	4. E: embedding matrix
	5. ntwk_type: string of either: `rnn`, `lstm` or `gru`
"""
function reuters_test_error(network :: Network.network, features, labels, E :: Matrix, ntwk_type :: String)
	#Verify prediction and true label are same size
	network.results != size(labels,2) && @error "Prediction and true label are different dimensions."

	N = size(features,1)
	num_correct = 0
	x0 = ntwk_type == "lstm" ? 2*network.hid_dim : network.hid_dim

	for i=1:N
		U, label = Helpers.embed(features[i,:],E), labels[i,:]
		U = vcat(collect(Iterators.flatten(U')), zeros(network.hid_dim))
		X = Network.evaluate(network, U)[end-network.results+1:end]
		Z = exp.(X .- maximum(X))
		pred = Z / sum(Z)
		findmax(pred)[2] == findmax(label)[2] ? num_correct += 1 : num_correct += 0 ;
   end
   return 1.0 - num_correct/N
end
=#

"""
Description:
	Returns indices of X for specified network type
Input:
	ntwk_type :: String ("rnn", "lstm", "gru")
	ntwk :: Network.network
"""
function ntwkX_ind(ntwk :: Network.network, ntwk_type :: String)
	λ_ind = ntwk_type == "rnn" ?
			RNN.rnnXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length) :
			ntwk_type == "lstm" ?
					LSTM.lstmXind(ntwk.features-2*ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length) :
					GRU.gruXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length)
	return λ_ind
end

function init_adjoints(Λ, hid_dim)
	L, N = size(Λ)
	ind = 1:hid_dim
	λ = zeros(hid_dim,N)
	for i=1:N
		λ[:,i] = Λ[ind,i]
	end
	return λ
end

function output_adjoints(Λ, hid_dim)
	L, N = size(Λ)
	ind = L-hid_dim+1:L
	λ = zeros(hid_dim,N)
	for i=1:N
		λ[:,i] = Λ[ind,i]
	end
	return λ
end
"""
Description:
	Returns specified network type architecture as ntwk, partition and syncFlag
Input:
	ntwk_type :: String ("rnn", "lstm", "gru")
	ntwk :: Network.network
"""
function specify_ntwk(ntwk_type :: String, input_dim :: Int64, output_dim :: Int64, hid_dim :: Int64, fc_dim :: Int64, seq_length :: Int64, seed :: Int64)

	ntwk, partition, syncFlag =	ntwk_type == "rnn" ?
		RNN.specifyRNN(input_dim+hid_dim, output_dim, hid_dim, fc_dim, seq_length, seed) :
		NTWK_TYPE == "lstm" ?
			LSTM.specifyLSTM(input_dim+2*hid_dim, output_dim, hid_dim, fc_dim, seq_length, seed) :
			GRU.specifyGRU(input_dim+hid_dim, output_dim, dim_dim, fc_dim, seq_length, seed)
	return ntwk, partition, syncFlag
end


"""
Load MNIST test set
"""
function load_mnist_test()
	# load MNIST data
	test_x,  test_y  = MNIST.testdata()
	# reshape features and one-hot encode labels
	test_x = map(i -> collect(Iterators.flatten(test_x[:,:,i])), 1:size(test_x,3))
	test_y = Helpers.one_hot(test_y)
	return test_x, test_y
end
"""
************** Optimizer/Update Functions **************
"""

"""
Initializes `m` and `v` for adam update
"""
function adam_init(ntwk :: Network.network)
	m = []
	v = []
	for i=1:length(ntwk.neurons)
		l_i = length(ntwk.neurons[i].β)
		append!(m, [zeros(l_i)])
		append!(v, [zeros(l_i)])
	end
	return m, v
end
function adam_init(ntwk :: Network.network, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
	m = []
	v = []
	for class in partition
		max_j = maximum(class)
		l_i = length(ntwk.neurons[max_j].β)
		append!(m, [zeros(l_i)])
		append!(v, [zeros(l_i)])
	end
	return m, v
end


"""
*********** Batch Functions ***************
"""

"""
Performs batch training
	- Updates network parameters with batch gradient (sum)
	- Returns batch statistics: F, G, Λ  (avg)
	- batch_x = [num_samples, seq_length, u_dim]
"""
function evaluate_batch(ntwk :: Network.network, batch_x, batch_y, loss :: Module, penalty :: Module, η :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, optimizer :: String, method :: String, ntwk_type :: String, m, v, i)
	#batch_x, batch_y = train_x[j], train_y[j]

	# evaluates entire batch -- returns batch sums
	b_grads, b_F, b_G, b_Λ = Helpers.batch_summary(ntwk, batch_x, batch_y, loss, penalty, syncFlag, partition, method, ntwk_type)

	# update parameters
	optimizer == "adam" ?
		adam_update!(ntwk, b_grads, η, syncFlag, partition, m, v, i) :
		update!(ntwk, b_grads, η, syncFlag, partition)

	return b_F/length(batch_x), b_G/length(batch_x), b_Λ/length(batch_x)
end

function evaluate_batch(ntwk :: Network.network, batch_x, batch_y, loss :: Module, penalty :: Module, η :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, optimizer :: String, method :: String, ntwk_type :: String, m, v, i, batch_size)
	#batch_x, batch_y = train_x[j], train_y[j]

	# evaluates entire batch -- returns batch sums
	b_grads, b_F, b_G, b_Λ = Helpers.batch_summary(ntwk, batch_x, batch_y, loss, penalty, syncFlag, partition, method, ntwk_type, batch_size)

	# update parameters
	optimizer == "adam" ?
		adam_update!(ntwk, b_grads, η, syncFlag, partition, m, v, i) :
		update!(ntwk, b_grads, η, syncFlag, partition)

	return b_F/length(batch_x), b_G/length(batch_x), b_Λ/length(batch_x)
end

"""
Description:
	Given a vector of `paramGrad` dictionaries, either
		i. returns the batch gradient
		ii. returns error with specified reason
Dependencies:
	- LinearAlgebra
Input:
	1. grad_array : A vector of length batch size that returns either the batch
					sum or batch average
	2. stat (default="sum") : batch statistic specifying either "sum" or "avg"

"""

function batch_grad(grad_array :: Vector; stat = "sum")
	(stat ∉ ["sum", "avg"]) && @error "Batch statistic must either be `sum` or `avg`."
	if stat == "sum"
		return Dict(k => sum(map(i -> grad_array[i][k], 1:length(grad_array))) for k in keys(grad_array[1]))
	else
		return Dict(k => sum(map(i -> grad_array[i][k], 1:length(grad_array))) / length(grad_array) for k in keys(grad_array[1]))
	end
end

"""
Description:
    Returns batch gradient
Input:
    1. ntwk :  Network
    2. batch_x : features of batch i
	3. batch_y : labels of batch i
    4. loss :: cost function
	- batch_x = [num_samples, seq_length, u_dim]
	- batch_y = [num_samples, output_dim]
"""
function batch_summary(ntwk :: Network.network, batch_x, batch_y, loss :: Module, syncFlag :: Bool, partition)
    N = size(batch_y,1)
    F = 0.0
    grads = []
    for i=1:N
        U, label = vcat(batch_x[i], zeros(ntwk.hid_dim)), batch_y[i,:]
        X = Network.evaluate(ntwk, U)
        #F += loss.evaluate(label, X, ntwk.results)
        if syncFlag
            push!(grads, paramGrad(loss, ntwk, U, label, syncFlag, partition))
		else
            push!(grads, paramGrad(loss, ntwk, U, label))
		end
    end
    ∇batch = Helpers.batch_grad(grads)
    return ∇batch, F / N
end
function batch_summary(ntwk :: Network.network, batch_x, batch_y, loss :: Module, penalty :: Module, syncFlag :: Bool, partition, method :: String, ntwk_type :: String)

	b_F = 0.0
	b_G = 0.0
	b_Λ = zeros(size(ntwk.graph,1))

	N = size(batch_y,1)
	grads = []
	for i=1:N
		U, label = vcat(batch_x[i], zeros(ntwk.hid_dim)), batch_y[i,:]
		X = Network.evaluate(ntwk, U)
		b_Λ += Network.adjoint(ntwk, X, label, loss)
		b_F += loss.evaluate(label, X, ntwk.results)
		b_G += penalty.evaluate(b_Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type)
		if syncFlag
			method == "adjoint" ?
				push!(grads, paramGrad(loss, ntwk, U, label, syncFlag, partition)) :
				push!(grads, paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type))
		else
			method == "adjoint" ?
				push!(grads, paramGrad(loss, ntwk, U, label)) :
				push!(grads, paramGrad(loss, penalty, ntwk, U, label))
		end
	end
	∇batch = Helpers.batch_grad(grads)
	return ∇batch, b_F, b_G, b_Λ # returns batch sums
end
function batch_summary(ntwk :: Network.network, batch_x, batch_y, loss :: Module, penalty :: Module, syncFlag :: Bool, partition, method :: String, ntwk_type :: String, batch_size :: Int64)
	(batch_size != size(batch_x,1)) && @error "batch does not match specified batch size."

	b_F = 0.0
	b_G = 0.0
	b_Λ = zeros(size(ntwk.graph,1))

	N = batch_size
	grads = []
	for i=1:N
		U, label = collect(Iterators.flatten(vcat(batch_x[i,:,:], zeros(ntwk.hid_dim)))), batch_y[i,:]
		X = Network.evaluate(ntwk, U)
		λ = Network.adjoint(ntwk, X, label, loss)
		b_Λ += λ
		b_F += loss.evaluate(label, X, ntwk.results)
		b_G += penalty.evaluate(λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type)
		if syncFlag
			method == "adjoint" ?
				push!(grads, paramGrad(loss, ntwk, U, label, syncFlag, partition)) :
				push!(grads, paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type))
		else
			method == "adjoint" ?
				push!(grads, paramGrad(loss, ntwk, U, label)) :
				push!(grads, paramGrad(loss, penalty, ntwk, U, label))
		end
	end

	∇batch = Helpers.batch_grad(grads)
	return ∇batch, b_F, b_G, b_Λ # returns batch sums
end

"""
Description:
	Given a dataset returns the `batched` set.
Dependencies:
	- MLDatasets
	- Random
Input:
	1. dataset (String) : name indicator of dataset in MLDatasets package
	2. batch_size (Integer): the number of samples used per batch
	3. seed
"""

function batch(dataset :: String, batch_size :: Int64, seed :: Int64)
	# Note:
		# Need to add functionality for when number of samples
		# is not divisible by batch size.
	(dataset != "mnist") && @error "Dataset not currently available."
	# load and reshape train & test data
	train_x, train_y = MNIST.traindata()
	train_x = map(i -> collect(Iterators.flatten(train_x[:,:,i])), 1:size(train_x,3))
	train_y = Helpers.one_hot(train_y)
	# permute training set
	Random.seed!(seed)
	perm = shuffle(1:length(train_x))
	train_x = train_x[perm]
	train_y = train_y[perm,:]

	# generate batch sequence over train_x dimensions
	batch_ind = collect(0:batch_size:length(train_x))[2:end]

	b_train_x = Dict() # (batch #, sample i in batch, feature size)
	b_train_y = Dict() # (batch #, sample i in batch)

	for i=1:length(batch_ind)
		if i == 1
			push!(b_train_x, i => train_x[1:batch_ind[i]])
			push!(b_train_y, i => train_y[1:batch_ind[i],:])
		elseif i != length(batch_ind)
			push!(b_train_x, i => train_x[batch_ind[i-1]+1:batch_ind[i]])
			push!(b_train_y, i => train_y[batch_ind[i-1]+1:batch_ind[i],:])
		else # i == length(batch_ind) # ie last batch

			push!(b_train_x, i => train_x[batch_ind[i]:end])
			push!(b_train_y, i => train_y[batch_ind[i]:end,:])
		end
	end

	return (b_train_x, b_train_y)
end


"""
Verifies/flags numerical stability during training
"""

function verifyNumericalStability(F :: Float64, G :: Float64)
    flag = true
    if isnan(F)
        printstyled("*************** Training Stopped: NaNs Produced. ***************\n\n", color=:red)
        flag = false
    elseif G == Inf
        printstyled("*************** Training Stopped: Inf Produced. ***************\n\n", color=:red)
        flag = false
    end
    return flag
end

"""
Description:
	Returns one-hot labels (matrix) for multi-class classification problems
input:
	label :: vector (e.g. label = [1,2,3])
output:
	label matrix (e.g. label = [[1,0,0], [0,1,0], [0,0,1]])
"""
function one_hot(label :: Vector)
    sort_label = sort(unique(label))
    L = length(unique(label))
    one_hot = zeros(Float64,length(label), L)
    for i=1:length(label)
        pos = findall(x -> x == label[i], sort_label)[]
        one_hot[i, pos] = 1.0
    end
    return one_hot
end

"""
Description:
	Returns initial error and F (cost)
"""

function mnist_init_error(ntwk :: Network.network, test_x, test_y, loss :: Module, ntwk_type :: String)
	init_F, init_error = Helpers.mnist_test_error(ntwk, test_x, test_y, loss, ntwk_type)
	printstyled("Initial Test Error:\t$(init_error)\n", color=:cyan)
	printstyled("Initial Test F:\t\t$(init_F)\n\n", color=:cyan)
end

"""
Returns test error for MNIST
"""
function mnist_test_error(network, features, labels, loss :: Module, ntwk_type :: String)
	"""
	Compute test error over 10K MNIST samples
	"""
	#Verify prediction and true label are same size
	network.results != size(labels, 2) && @error "Prediction and true label are different dimensions."

	N = length(features)
	F = 0.0
	correct = zeros(N)
	x0 = ntwk_type == "lstm" ? 2*network.hid_dim : network.hid_dim

	for i=1:N
		U, label = features[i], labels[i,:]
		# append initial hidden state (zero vector)
		U = vcat(U, zeros(x0))
		X = Network.evaluate(network, U)[end-network.results+1:end]
		F += loss.evaluate(label, X, network.results)
		Z = exp.(X .- maximum(X))
		pred = Z / sum(Z)
		correct[i] = findmax(pred)[2] == findmax(label)[2] ? 1.0 : 0.0 ;
   end
   error = 1.0 - sum(correct)/N
   return F/N, error
end

function mnist_test_error(network, features, labels, loss :: Module, penalty :: Module, ntwk_type :: String)
	"""
	Compute test error over 10K MNIST samples
	"""
	#Verify prediction and true label are same size
	network.results != size(labels, 2) && @error "Prediction and true label are different dimensions."

	N = length(features)
	F = 0.0
	G = 0.0
	Λ = zeros(size(network.graph,1))
	correct = zeros(N)
	x0 = ntwk_type == "lstm" ? 2*network.hid_dim : network.hid_dim

	for i=1:N
		U, label = features[i], labels[i,:]
		# append initial hidden state (zero vector)
		U = vcat(U, zeros(x0))
		X = Network.evaluate(network, U)
		λ = Network.adjoint(network, X, label, loss)
		F += loss.evaluate(label, X, network.results)
		Λ += λ
		G += penalty.evaluate(λ, network.features, network.hid_dim, network.seq_length, ntwk_type)
		X = X[end-network.results+1:end]
		Z = exp.(X .- maximum(X))
		pred = Z / sum(Z)
		correct[i] = findmax(pred)[2] == findmax(label)[2] ? 1.0 : 0.0 ;

		Z = exp.(X .- maximum(X))
		pred = Z / sum(Z)
		correct[i] = findmax(pred)[2] == findmax(label)[2] ? 1.0 : 0.0 ;
   end
   e = 1.0 - sum(correct)/N
   return F/N, G/N, Λ/N, e
end

"""
Description:
	Evaluates training set for binary classificatin tasks
Returns:
	1. F - test error
	2. E - test classication error
"""
function eval_binary_loss(ntwk :: Network.network, test_x, test_y, loss :: Module, penalty :: Module, ntwk_type :: String)
	N = size(test_x,1)

	F = 0.0
	G = 0.0
	Λ = zeros(size(ntwk.graph,1))
	e = zeros(N)

	for i=1:N
		U, label = collect(Iterators.flatten(vcat(test_x[i,:,:], zeros(ntwk.hid_dim)))), test_y[i,:]
		X = Network.evaluate(ntwk, U)
		λ = Network.adjoint(ntwk, X, label, loss)
		Λ += λ
		F += loss.evaluate(label, X, ntwk.results)
		G += penalty.evaluate(λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type)
		pred = X[end] > 0.5 ? 1.0 : 0.0
		e[i] = pred == label[] ? 1.0 : 0.0
	end
	return F/N, G/N, Λ/N, 1.0-sum(e)/N
end

"""
**************SAVING NETWORKS/PARAMETERS*****************
"""


"""
Description:
	Saves parameters and hiddens state of network
		- writes an arry of length 3
		- first element in array = ntwk.Par_Hid
		- second element in array = ntwk.Par_Out
		- third element in array = ntwk.Sta_Hid
Output:
	- 'filename.jld' is stored in 'directory'
	- used for old code (RNN_Proposal)
"""
function save_network(ntwk :: Network.network, file_name :: String, directory :: String)
   par_hid = ntwk.Par_Hid
   par_out = ntwk.Par_Out
   sta_hid = ntwk.Sta_Hid
   ntwk_str = [par_hid, par_out, sta_hid]
   # write to file -- requires pkg JLD
   directory = "/" * directory * "/"
   file_name = file_name * ".jld"
   name = directory * file_name
   save(name, "model", ntwk_str)
   return
end

"""
*************TRANSFERING PARAMETERS FROM PYTHON TO JULIA****************
"""

"""
Set rnn ntwk parameters to saved tf-keras params
"""
function set_rnn_params(ntwk :: Network.network, partition :: Vector{Vector{Int64}}, w_rec, w_rec_bias, w_out, w_out_bias)

	# recurrent paramaters
	for i=1:ntwk.hid_dim #20
		for j=1:ntwk.seq_length # 28
			id = partition[i][j]
			ntwk.neurons[id].β = vcat(w_rec_bias[i], w_rec[:,i])
		end
	end
	ct = 1
	for i=ntwk.hid_dim+1:length(partition)
		id = partition[i][]
		ntwk.neurons[id].β = vcat( w_out_bias[ct], w_out[:,ct])
		ct += 1
	end
	return ntwk
end

"""
Set lstm ntwk parameters to saved tf-keras params
"""
function set_lstm_params(ntwk :: Network.network, partition :: Vector{Vector{Int64}}, w_f, w_i, w_c, w_o, w_out)
	hid_dim = ntwk.hid_dim
	seq_length = ntwk.seq_length
	part_f = partition[1:hid_dim]
	part_i = partition[hid_dim+1:2*hid_dim]
	part_o = partition[2*hid_dim+1:3*hid_dim]
	part_c = partition[3*hid_dim+1:4*hid_dim]

	for i=1:hid_dim
		for j=1:seq_length
			f_id = part_f[i][j]
			i_id = part_i[i][j]
			o_id = part_o[i][j]
			c_id = part_c[i][j]
			ntwk.neurons[f_id].β = w_f[:,i]
			ntwk.neurons[i_id].β = w_i[:,i]
			ntwk.neurons[o_id].β = w_o[:,i]
			ntwk.neurons[c_id].β = w_c[:,i]
		end
	end

	part_out = partition[end-ntwk.results+1:end]
	for i=1:ntwk.results#length(partition)-ntwk.results+1:length(partition)
		id = part_out[i][]
		ntwk.neurons[id].β = w_out[:,i]
	end
	return ntwk
end

"""
Description:
	For a given test set, returns the gradients with respect to X (input)
Input:
	1. ntwk : Network.network
	2. loss : Module
	3. penalty : Module
	4. test_x : features
	5. test_y : labels
	6. ntwk_type : String indicating type of recurrent network
Returns:
	1. gradients with respect to input for each sample
	2. F : Test error
	3. G : Test G
	4. Λ : adjoints for every sample
"""
function inputGradients(ntwk :: Network.network, loss :: Module, penalty :: Module, test_x, test_y, method :: String, ntwk_type :: String)
	N = length(test_x)
	x0 = ntwk_type == "lstm" ? 2*ntwk.hid_dim : ntwk.hid_dim
	F = 0.0
	G = 0.0
	Λ = zeros(size(ntwk.graph,1), N)
	grads = []
	for i=1:N
		U, label = test_x[i], test_y[i,:]
		# append initial hidden state (zero vector)
		U = vcat(U, zeros(x0))
		X = Network.evaluate(ntwk, U)
		λ = Network.adjoint(ntwk, X, label, loss)
		F += loss.evaluate(label, X, ntwk.results)
		Λ[:,i] = λ
		G += penalty.evaluate(λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type)
		method == "adjoint" ?
			push!(grads, inputGrad(loss, ntwk, U, label)) :
			push!(grads, inputGrad(loss, penalty, ntwk, U, label, ntwk_type))
	end
	return F/N, G/N, Λ, grads
end

"""
Description:
	Given a dictionary of neuron gradients wrt input, returns the gradients with respect to the input U
Note:
	This is for a single evaluation of U

"""
function reform_single_input_grad(gradient :: Dict{Int64,Vector{Float64}}, ntwk :: Network.network, partition :: Vector{Vector{Int64}})
	u_dim = convert(Int64, (ntwk.features-ntwk.hid_dim)/ntwk.seq_length)
	rec_part = partition[1:ntwk.hid_dim]
	dU = zeros(u_dim, ntwk.seq_length)
	for i=1:ntwk.seq_length
		for j=1:length(rec_part)
			nrn = rec_part[j][i]
			dU[:,i] += gradient[nrn][1:u_dim]
		end
	end
	dU = collect(Iterators.flatten(dU))
	(ntwk.features-ntwk.hid_dim != length(dU)) && @error "Gradient wrt input does not match input dimensions."
	return dU
end
"""
Description:
	Given a network and a vector of gradients, returns the gradients with respect to the input
Returns:
	A matrix of size (features length, num_samples)
"""
function all_input_grads(ntwk :: Network.network, grads :: Vector, partition :: Vector{Vector{Int64}})
	N = length(grads)
	dU = zeros(ntwk.features-ntwk.hid_dim, N)
	for i=1:N
		gradient = grads[i]
		dU[:,i] = reform_single_input_grad(gradient, ntwk, partition)
	end
	return dU
end

function all_input_grads(ntwk :: Network.network, loss :: Module, penalty :: Module, test_x, test_y, partition :: Vector{Vector{Int64}}, method :: String, ntwk_type :: String)
	_, _, _, grads = inputGradients(ntwk, loss, penalty, test_x, test_y, method, ntwk_type)
	N = length(grads)
	dU = zeros(ntwk.features-ntwk.hid_dim, N)
	for i=1:N
		gradient = grads[i]
		dU[:,i] = reform_single_input_grad(gradient, ntwk, partition)
	end
	return dU
end
"""
Description:
	Given a matrix (2d array) of feature gradients [gradient, N samples] returns
	the matrix of gradients corresponding to a specified sequence element (ie uι)
"""
function input_element_grad(grads :: Array{Float64,2}, ntwk :: Network.network, element :: Int64)
	m, N = size(grads)
	u_dim = convert(Int64, m / ntwk.seq_length)
	m_mat = reshape(1:m, (ntwk.seq_length,u_dim))
	e_i = m_mat[:,element]
	dU = zeros(u_dim, size(grads,2))
	for i=1:N
		dU[:,i] = grads[:,i][e_i]
	end
	return dU
end
function input_element_grad(ntwk :: Network.network, loss :: Module, penalty :: Module, test_x, test_y, partition :: Vector{Vector{Int64}}, method :: String, ntwk_type :: String, element :: Int64)
	grads = all_input_grads(ntwk, loss, penalty, test_x, test_y, partition, method, ntwk_type)
	m, N = size(grads)
	u_dim = convert(Int64, m / ntwk.seq_length)
	m_mat = reshape(1:m, (ntwk.seq_length,u_dim))
	e_i = m_mat[:,element]
	dU = zeros(u_dim, size(grads,2))
	for i=1:N
		dU[:,i] = grads[:,i][e_i]
	end
	return dU
end
"""
Description:
	Given a gradient matrix of [∇ui,num samples], performs PCA and returns the first principal direction
	with respect to ∇ui
Dependencies:
	MultivariateStats
"""
function principal_direction(G :: Array{Float64,2})
	P = fit(PCA, G, method=:cov)
	d = projection(P)[:,1]
	return d
end
"""
Description:
	Given feature dim, sequence length and sequence position; returns the indicies
	of the given feature that specify the elements of the provided sequence position
"""
function sequence_position(feat_dim :: Int64, seq_length :: Int64, pos :: Int64)
	u_dim = convert(Int64, feat_dim/seq_length)
	M = reshape(collect(1:feat_dim), u_dim, seq_length)
	id = convert.(Int64, M[:,pos])
	return id
end

"""
Description:
	Given a feature, sequence position and a direction vector, returns the perturbed feature
	in the direction of the provided direction vector
"""
function perturb_input(U, pos :: Int64, d :: Array{Float64,1}, seq_length :: Int64, ϵ :: Float64)
	U = convert.(Float64,U)
	feat_dim = length(U)
	u_pos = sequence_position(feat_dim, seq_length, pos)
	(length(u_pos) != length(d)) && @error "sequence element and direction vector are not of equal dimension"
	U[u_pos] += d*ϵ
	return U
end
"""
Description:
	Given a dataset, sequence position and a direction vector, returns the perturbed dataset
	in the direction of the provided direction vector
"""
function perturb_dataset(features, pos :: Int64, d :: Array{Float64,1}, seq_length :: Int64, ϵ :: Float64)
	N = length(features)
	U_pert = []
	for i=1:N
		U = length(features[i]) == 1 ? features[i][] : features[i]
		#U = convert.(Float64, features[i])
		uι = perturb_input(U, pos, d, seq_length, ϵ)
		push!(U_pert, uι)
	end
	(length(U_pert) != length(features)) && @error "Perturbed and original dataset dimensions do not match"
	return U_pert
end

"""
Description:
	Generate training batch for toy experiment
Return:
	U 		: (batch_size,seq_length,u_dim)
	label 	: (batch_size,1)
Note: indictor should reflect position+1 due to python indexing difference
"""
function generate_batch(batch_size :: Int64, seq_length :: Int64, indicator :: Int64)
    U = zeros(batch_size,seq_length,1)
    label = zeros(batch_size,1)
    for i=1:batch_size
        u = randn(seq_length,1)
        U[i,:,:] = u
        label[i] = u[indicator] > 0 ? 1.0 : 0.0
	end
    return U, label
end
"""
Description:
	Generates dataset for toy experiment
Return:
	feat	: (N,seq_length,u_dim)
	label 	: (N,1)
"""
function generate_dataset(N :: Int64, seq_length :: Int64, indicator :: Int64)
    feat, label = generate_batch(N, seq_length, indicator)
    return feat, label
end

end # end module
