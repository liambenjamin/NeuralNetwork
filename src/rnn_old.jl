# RNN Module -- Functions for constructing RNN architecture
# Author: Liam Johnston
# Date: August 24, 2020


module RNN

using Random, Statistics, SparseArrays, Main.Neurons, Main.Network, Main.Partition

"""
Returns vector of length `seq_length` where the i^{th} entry is a 2-dim vector containing
the starting and ending indices for time step i, respectively.
- Note: returns indices of X corresponding to uι
"""
function getFeatureIndex(feat_dim, seq_length)
    u_l = convert(Int64, feat_dim/seq_length)
    ft_ind = map(i -> hcat(i, i+u_l-1), 1:u_l:feat_dim)
    return ft_ind
end

"""
Returns state indices of X-vector corresponding where each gating unit is stored
consecutively and of length `hid_dim`
- Preserves hidden dimension.
"""
function getStateIndex(feat_dim, hid_dim, seq_length, output_dim)
    x_prev = map(i -> convert(Int64,i), feat_dim+1:feat_dim+hid_dim) #init at x₀

    ind_end = x_prev[end] #holds current place in X

    state_ind = []

	# rnn cells
    for i=1:seq_length
		x_nxt = zeros(hid_dim)
		for j=1:hid_dim
			append!(state_ind, [convert.(Int64, x_prev)])
			ind_end += 1
			x_nxt[j] = ind_end
		end
		#update x_prev
		x_prev = x_nxt
    end #end rnn cells

	# rnn end -> fc
	x_nxt = zeros(output_dim)
	for i=1:output_dim
		append!(state_ind, [convert.(Int64, x_prev)])
		ind_end += 1
		x_nxt[i] = ind_end
	end
	# fc -> softmax
	append!(state_ind, [convert.(Int64, x_nxt)])
	# softmax -> prediction
	append!(state_ind, [convert.(Int64, map(i -> i, x_nxt[end]+1:x_nxt[end]+output_dim))])

    return state_ind
end

"""
Returns initialized neurons for RNN of specified dimensions
"""
function rnnNeurons(feat_dim, hid_dim, seq_length, output_dim, seed; kwargs...)
    Random.seed!(seed)
    neurons = Vector{neuron}(undef, seq_length * hid_dim + output_dim + 1)
    u_l = convert(Int64, feat_dim / seq_length)
	id = 1
	for i=1:seq_length
		for j=1:hid_dim
			neurons[id] = sigmoid.init(id, u_l+hid_dim, u_l+hid_dim+1)
			id += 1
		end
	end

    for i=1:output_dim
        neurons[id] = sigmoid.init(id, hid_dim, hid_dim+1)
		id += 1
    end
    neurons[end] = softmax.init(length(neurons), output_dim, 1; met = kwargs)
    return neurons
end

"""
Returns RNN graph index
- Combines feature and state indices
"""

function graphIndex(feat_ind, state_ind, hid_dim, output_dim)
	graph_index = []
	ct = 1
	# RNN neurons
	for i=1:length(feat_ind)
		for j=1:hid_dim
			append!(graph_index, [vcat(feat_ind[i][1]:feat_ind[i][2], state_ind[ct])])
			ct += 1
		end
	end

	for i=ct:length(state_ind)
		append!(graph_index, [state_ind[i]])
	end

	return graph_index
end


function graphRNN(feat_dim, hid_dim, seq_length, output_dim, seed)
    neurons = rnnNeurons(feat_dim, hid_dim, seq_length, output_dim, seed)

    # feature indices
    feat_ind = getFeatureIndex(feat_dim, seq_length)
    # state indices
    state_ind = getStateIndex(feat_dim, hid_dim, seq_length, output_dim)
	# graph index
	g_index = graphIndex(feat_ind, state_ind, hid_dim, output_dim)

    rowInd = [] #indices of input to neuron i
    colInd = [] #neuron i repeated length of input indices

	for i=1:length(g_index)
		append!(rowInd, g_index[i])
		append!(colInd, i * ones(Int64, length(g_index[i])))
	end

    return neurons, rowInd, colInd
end


"""
Returns tuple of neuron partitioning for sharing weights (RNN)
Options:
- num_neurons: total number of neurons in network
- hid_dim: number of neurons in a single hidden layer
"""
function getRNNPartition(num_neurons, hid_dim :: Int64, seq_length :: Int64, output_dim :: Int64)
    R = 1:(seq_length*hid_dim)
    part_matrix = reshape(R, hid_dim, seq_length)
    L = size(part_matrix, 1)
    partition = Array{Array{Int64,1}}(undef,L+output_dim+1)
    partition[1:L] = map(i -> convert.(Int64,part_matrix[i,:]), 1:L)
    partition[L+1:end] = map(i -> [convert(Int64,i)], (L*size(part_matrix,2)+1):num_neurons)
    return partition
end


"""
Specifies graph for RNN
"""
function specifyRNN(inp_dim :: Int64, output_dim :: Int64, hid_dim :: Int64, fc_dim :: Int64, seq_length :: Int64, seed :: Int64)
	feat_dim = inp_dim - hid_dim
	# specify graph
    neurons, rowInd, colInd = graphRNN(feat_dim, hid_dim, seq_length, output_dim, seed)
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd,colInd,vals)
    ntwk = Network.network(neurons, inp_dim, output_dim, hid_dim, fc_dim, seq_length)
    Network.graph!(ntwk,graph)

    # partition neurons to share weights
    partition = getRNNPartition(length(ntwk.neurons), hid_dim, seq_length, output_dim)
    Partition.synchronizeParameters!(ntwk, partition)

    # verify partition
    syncFlag = Partition.verifyPartition(ntwk, partition)

    return ntwk, partition, syncFlag
end

"""
Returns test error for synthetic experiment
"""
function test_error(ntwk, info_layer :: Int64, loss :: Module)
	error = 0.0
	N = 1000
    F = 0.0

	for i=1:N
		U, label = vcat(collect(Iterators.flatten(randn(1,10))), zeros(ntwk.hid_dim)), zeros(ntwk.results)
		sum(U) > 0 ?
			label[1] = 1.0 :
			label[2] = 1.0
		X = Network.evaluate(ntwk, U)
		pred = X[end-ntwk.results+1:end]
		pred_pos = findall(x -> x == maximum(pred), pred)
		lab_pos = findall(x -> x == maximum(label), label)
		# compute classification error
		pred_pos == lab_pos ?
			error = error :
			error += 1
        # Test F
        F += loss.evaluate(label, X, ntwk.results)
	end
	return error/N, F/N
end

"""
Returns test error for adding synthetic experiment
"""
function add_test_error(ntwk, loss :: Module)
	error = 0.0
	N = 2500
    F = 0.0

	for i=1:N
		U, label = vcat(rand(10), zeros(ntwk.hid_dim)), zeros(ntwk.results)
		U[1:10] = map(i -> U[i] > 0.5 ? 1.0 : 0.0, 1:10)
		label[convert(Int64, (sum(U)+1))] = 1.0
		X = Network.evaluate(ntwk, U)
		pred = X[end-ntwk.results+1:end]
		pred_pos = findall(x -> x == maximum(pred), pred)
		lab_pos = findall(x -> x == maximum(label), label)
		# compute classification error
		pred_pos == lab_pos ?
			error = error :
			error += 1
        # Test F
        F += loss.evaluate(label, X, ntwk.results)
	end
	return error/N, F/N
end

"""
Returns classification error rate for network (RNN)
	Assumes:
			- size(test_features) = (num_samples, input_length, time_steps)
			- size(test_labels) = (num_samples, <one-hot encoded labels by row>)
			- ``binary``: indicates binary classification task where input takes values yι ∈ {0,1}
"""
function classification_error(network, test_features, test_labels, binary :: Bool)
    #Verify prediction and true label are same size
	if binary == true
		network.results != 2 && @error "Prediction and true label are different dimensions."
	else
    	network.results != size(test_labels, 2) && @error "Prediction and true label are different dimensions."
	end

    hid_state = network.hid_dim
    num_correct = zeros(size(test_features,1))
    for i=1:size(test_features,1)
		U, label = collect(Iterators.flatten(test_features[i,:,:])), zeros(network.results)
		test_labels[i] > 0 ? label[1] = 1.0 : label[2] = 1.0
		# append initial hidden state (zero vector)
        append!(U, zeros(hid_state))
        pred = Network.evaluate(network, U)[end-network.results+1:end]
        Y = findmax(pred)[2]
        num_correct[i] = Y == findmax(label)[2] ? 1 : 0 ;
   end
   return 1.0 - (sum(num_correct) / size(test_features,1))
end

"""
Returns position of 'X' indices that correspond to hidden states
"""
function rnnXind(feat_dim, hid_dim, seq_length)
    x0 = map(i -> convert(Int64,i), feat_dim+1:feat_dim+hid_dim) #init at x₀
	X_pos = x0 # length `hid_dim`

	ind_end = x0[end] #holds current place in X
	x_ind = hid_dim+1 #indexes position of X_pos

	# rnn cells
    for i=1:seq_length
		x_nxt = zeros(hid_dim)
		for j=1:hid_dim
			ind_end += 1
			x_nxt[j] = ind_end
			append!(X_pos, ind_end)
		end
		#update x_prev
		x_prev = x_nxt
    end #end rnn cells

	return X_pos
end



end # end RNN module
