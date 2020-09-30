# GRU Module -- Functions for constructing GRU architecture
# Author: Liam Johnston
# Date: August 27, 2020

module GRU

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
	- Preserves hidden dimension
"""
function getStateIndex(feat_dim, hid_dim, seq_length, output_dim)
    x_prev = map(i -> convert(Int64,i), feat_dim+1:feat_dim+hid_dim) #init at x₀
    ind_end = x_prev[end] #holds current place in X

    state_ind = []

	# gru cells
    for i=1:seq_length
		gru_mat = zeros(hid_dim,5) #columns of matrix correspond to Z,R,I,H,X position in `X-vector`
		gru_mat[:,end] = x_prev
		for j=1:5
			if j <= 2
				# Z, R
				for k=1:hid_dim
					append!(state_ind, [convert.(Int64,x_prev)])
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			elseif j == 3
				# I
				for k=1:hid_dim
					append!(state_ind, [convert.(Int64,vcat(gru_mat[k,2], gru_mat[k,end]))]) # r ∘ x_prev
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			elseif j == 4
				# H
				for k=1:hid_dim
					append!(state_ind, [convert.(Int64,gru_mat[:,3])]) # i
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			else
				# X
				for k=1:hid_dim
					append!(state_ind, [convert.(Int64, vcat(gru_mat[k,1],gru_mat[k,end],gru_mat[k,4]))])
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			end
		end
		#update x_prev
		x_prev = gru_mat[:,end]
    end #end gru cells

	# gru end -> fc
	x_nxt = zeros(output_dim)
	for i=1:output_dim
		append!(state_ind, [convert.(Int64, x_prev)])
		ind_end += 1
		x_nxt[i] = ind_end
	end

	# fc -> prediction
	append!(state_ind, [convert.(Int64, map(i -> i, x_nxt))])

    return state_ind
end

"""
Returns initialized neurons for GRU of specified dimensions
"""
function gruNeurons(feat_dim, hid_dim, seq_length, output_dim, seed)
    Random.seed!(seed)
    neurons = Vector{neuron}(undef, seq_length * hid_dim*5 + output_dim)
    u_l = convert(Int64, feat_dim / seq_length)
	id = 1
	for i=1:seq_length
		for j=1:5
			for k=1:hid_dim
				if j <= 2
					neurons[id] = sigmoid.init(id, u_l+hid_dim, u_l+hid_dim+1)
					id += 1
				elseif j == 3
					neurons[id] = hadamardCellState.init(id, 2, 3)
					id += 1
				elseif j == 4
					neurons[id] = hypertan.init(id, u_l+hid_dim, u_l+hid_dim+1)
					id += 1
				else
					neurons[id] = gruHiddenState.init(id, 3, 3+1)
					id += 1
				end
			end
		end
	end

	neurons[id:end] = map(i -> sigmoid.init(i, hid_dim, hid_dim+1), id:length(neurons))

    return neurons
end

"""
Returns GRU graph index
- Combines feature and state indices
"""

function graphIndex(feat_ind, state_ind, hid_dim, output_dim)
	graph_index = []
	ct = 1
	# GRU neurons
	for i=1:length(feat_ind)
		for j=1:5
			for k=1:hid_dim
				if j in [1,2,4]
					append!(graph_index, [vcat(feat_ind[i][1]:feat_ind[i][2], state_ind[ct])])
					ct += 1
				else
					append!(graph_index, [state_ind[ct]])
					ct += 1
				end
			end
		end
	end
	for i=ct:length(state_ind)
		append!(graph_index, [state_ind[i]])
	end

	return graph_index
end


function graphGRU(feat_dim, hid_dim, seq_length, output_dim, seed)
    neurons = gruNeurons(feat_dim, hid_dim, seq_length, output_dim, seed)

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
Returns tuple of neuron partitioning for sharing weights (GRU)
Options:
- num_neurons: total number of neurons in network
- hid_dim: number of neurons in a single hidden layer
"""
function getGRUPartition(num_neurons, hid_dim :: Int64, seq_length :: Int64, output_dim :: Int64)
    R = 1:(seq_length*hid_dim*5)
    part_matrix = reshape(R, hid_dim*5, seq_length)
    L = size(part_matrix, 1)
    partition = Array{Array{Int64,1}}(undef,L+output_dim)
    partition[1:L] = map(i -> convert.(Int64,part_matrix[i,:]), 1:L)
    partition[L+1:end] = map(i -> [convert(Int64,i)], (L*size(part_matrix,2)+1):num_neurons)
    return partition
end


"""
Specifies graph for GRU
"""
function specifyGRU(inp_dim :: Int64, output_dim :: Int64, hid_dim :: Int64, fc_dim :: Int64, seq_length :: Int64, seed :: Int64)
	feat_dim = inp_dim - hid_dim
	# specify graph
    neurons, rowInd, colInd = graphGRU(feat_dim, hid_dim, seq_length, output_dim, seed)
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd,colInd,vals)
    ntwk = Network.network(neurons, inp_dim, output_dim, hid_dim, fc_dim, seq_length)
    Network.graph!(ntwk,graph)

    # partition neurons to share weights
    partition = getGRUPartition(length(ntwk.neurons), hid_dim, seq_length, output_dim)
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
Returns position of 'X' indices that correspond to hidden states
"""
function gruXind(feat_dim, hid_dim, seq_length)
    x0 = map(i -> convert(Int64,i), feat_dim+1:feat_dim+hid_dim) #init at x₀
	X_pos = x0 # `length of hid_dim`

	ind_end = x0[end] #holds current place in X
	x_ind = hid_dim+1 # indexes position of X_pos

	# gru cells
    for i=1:seq_length
		gru_mat = zeros(hid_dim,5) #columns of matrix correspond to Z,R,I,H,X position in `X-vector`
		gru_mat[:,end] = x0
		for j=1:5
			if j <= 2
				# Z, R
				for k=1:hid_dim
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			elseif j == 3
				# I
				for k=1:hid_dim
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			elseif j == 4
				# H
				for k=1:hid_dim
					ind_end += 1
					gru_mat[k,j] = ind_end
				end
			else
				# X
				for k=1:hid_dim
					ind_end += 1
					gru_mat[k,j] = ind_end
					append!(X_pos, ind_end)
				end
			end
		end
		#update x_prev
		x0 = gru_mat[:,end]
	end #end gru cells

    return X_pos
end



end # end GRU module
