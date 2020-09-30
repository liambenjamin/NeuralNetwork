#Author: Liam Johnston
#Date: June 3, 2020
#Description: Functions used during training of neural networks


module Helpers

using Random, NPZ, CSV, DataFrames, Statistics, SparseArrays, Main.Neurons, Main.Network, Main.Partition

"""
Initializes `m` and `v` for adam update
"""
function adam_init(ntwk)
	m = []
	v = []
	for i=1:length(ntwk.neurons)
		l_i = length(ntwk.neurons[i].β)
		append!(m, [zeros(l_i)])
		append!(v, [zeros(l_i)])
	end
	return m, v
end
function adam_init(ntwk, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
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
Given array of `paramGrad` objects returns the batch gradient
"""

function batchGrad(grad_array :: Vector)
    return Dict(k => sum(map(i -> grad_array[i][k], 1:length(grad_array))) / length(grad_array) for k in keys(grad_array[1]))
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
Writes experiment to file
"""

function exp2file(file :: String, dataset :: String, name :: String, seed :: Int64, method :: String, lr :: Float64, hid_dim :: Int64, loss :: String, penalty :: String, output :: Dict)
    F = round(output["F"][end], digits=3)
    G = round(output["G"][end], digits=3)
    Error = round(output["Error"][end], digits=3)
    num_epochs = output["Epoch"][end]
    minF = round(minimum(output["F"]), digits=3)
    epochF = findall(x -> x == minimum(output["F"]), output["F"])
    minErr = round(minimum(output["Error"]), digits=3)
    epochErr = findall(x -> x == minimum(output["Error"]), output["Error"])
    minF = string("$(minF) / $(epochF)")
    minErr = string("$(minErr) / $(epochErr)")
    vals = DataFrame(Dataset=dataset, Name=name, Seed=seed, Method=method, LR=lr, hidDim=hid_dim, Epochs=num_epochs, Loss=loss, Penalty=penalty, F=F, G=G, Error =Error, MinF=minF, MinErr=minErr)
    CSV.write(file, vals, delim = ',', append=true)
end

"""
Implements learning rate scheduler
"""
function lr_scheduler(α :: Float64, epoch :: Int)
   if epoch < 25
      return α
   elseif epoch < 50
      return α * 0.5
   else
      return α * 0.5^2
   end
end

"""
Returns boolean indicating whether to decrease learning rate
- Determined by if overall objective decreased or not
"""
function adjust_lr(F :: Array{Float64,1}, G :: Array{Float64,1}, epoch :: Int64, method :: String)
    # don't evaluate if prior to epoch 2
    if epoch > 3
        # evaluate objective
        obj_prev = method == "adjoint" ? F[epoch-2] : F[epoch-2] + G[epoch-2]
        obj = method == "adjoint" ? F[epoch-1] : F[epoch-1] + G[epoch-1]
        flag = obj_prev < obj ? true : false
        return flag
    else
        return false
    end
end

"""
Function saves parameters and hidden state of network
- writes an arry of length 3
- first element in array = ntwk.Par_Hid
- second element in array = ntwk.Par_Out
- third element in array = ntwk.Sta_Hid
Output:
- 'filename.jld' is stored in 'directory'
- used for old code (RNN_Proposal)
"""
function save_network(ntwk, file_name :: String, directory :: String)
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
writes network parameters to file
- used for old code (RNN_Proposal)
"""
function save_network_params(network, method :: String, info_layer, init_number)
    hidden_pars = network.Par_Out
    output_pars = network.Par_Hid
    hid_name = string(method, "_network_", "hidden_params_", "info_$(info_layer)", "_", "init_", init_number, ".npz")
    out_name = string(method, "_network_", "output_params_", "info_$(info_layer)", "_", "init_", init_number, ".npz")
    npzwrite(hid_name, hidden_pars)
    npzwrite(out_name, output_pars)
end

"""
Returns classification error rate for network (Feed forward or RNN)
"""
function classification_error(network, test_features, test_labels)
    #Verify prediction and true label are same size
    network.results != size(test_labels, 2) && @error "Prediction and true label are different dimensions."

    hid_state = network.hid_dim
    num_correct = zeros(size(test_features, 3))
    for i=1:size(test_features,3)
       label, U = test_labels[i,:], convert(Array{Float64}, collect(Iterators.flatten(test_features[:,:,i])))
       # append initial hidden state (zero vector)
       append!(U, zeros(hid_state))
       pred = Network.evaluate(network, U)[end-network.results+1:end]
       Y = findmax(pred)[2]
       num_correct[i] = Y == findmax(label)[2] ? 1 : 0 ;
   end
   return 1.0 - (sum(num_correct) / size(test_features,3))
end

"""
Returns classification error rate for network training on urban-sound dataset
- Note: Need seperate functions as dimensions of input data are different
"""
function urban_classification_error(network, test_features, test_labels)
    #Verify prediction and true label are same size
    network.results != size(test_labels, 2) && @error "Prediction and true label are different dimensions."

    hid_state = network.hid_dim
    num_correct = zeros(size(test_features, 1))
    for i=1:size(test_features,1)
       U, label = collect(Iterators.flatten(test_features[i,:,:])), test_labels[i,:]
       # append initial hidden state (zero vector)
       append!(U, zeros(hid_state))
       pred = Network.evaluate(network, U)[end-network.results+1:end]
       Y = findmax(pred)[2]
       num_correct[i] = Y == findmax(label)[2] ? 1 : 0 ;
   end
   return 1.0 - (sum(num_correct) / size(test_features,1))
end


"""
Returns one-hot labels (matrix) for multi-class classification problems
input: label vector (e.g. label = [1,2,3])
output: label matrix (e.g. label = [[1,0,0], [0,1,0], [0,0,1]])
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
Writes epoch information to file
"""
function print_info(f :: String, i :: Int64, F :: Float64, G :: Float64, error :: Float64, Λ, time)
# print epoch summary information
   printstyled(f, "\n\n*****************************\nEpoch $(i) Summary Information\n*****************************\n\n", color=:magenta)

   printstyled(f, "\tEpoch Summary\n", color=:blue)
   printstyled(f, "\tF: $(F)\n", color=:blue)
   printstyled(f, "\tG: $(G)\n", color=:blue)
   printstyled(f, "\tClassification Error Rate: $(error)\n\n", color=:blue)

   printstyled(f, "\tInitial Layer Adjoint (norm): $(norm(Λ[785:794]))\n", color=:blue)
   printstyled(f, "\tOutput Layer Adjoint (norm): $(norm(Λ[end-19:end-10]))\n", color=:blue)

   printstyled(f, "\tEpoch time: $(time)\n")
end


"""
Returns test error for synthetic experiment
"""
function test_error(ntwk, info_layer :: Int64, loss :: Module)
	error = 0.0
	N = 1000
    F = 0.0

	for i=1:N
		U, label = randn(1,10), zeros(ntwk.results)
		if sum(U[:,info_layer]) > 0
			label[1] = 1.0
		else
			label[2] = 1.0
		end
		U = collect(Iterators.flatten(U))
		U = vcat(U, zeros(ntwk.hid_dim))
		X = Network.evaluate(ntwk, U)
		pred = X[end-ntwk.results+1:end]
		pred_pos = findall(x -> x == maximum(pred), pred)
		lab_pos = findall(x -> x == maximum(label), label)
		if pred_pos == lab_pos
			error = error
		else
			error = error + 1
		end
        # Test F
        F += loss.evaluate(label, X, ntwk.results)
	end
	return error/N, F/N
end

"""
Implements momentum update
options:
    - β: momentum hyperparameter (memory of past gradients, β=0 → no momentum)
    - v: past gradients, t-1
    - g: gradient at time t,t-1
"""
function momentumUpdate(β :: Float64, v , g)
    mom_grad = Dict( key => β*v[key] + g[key] for key in keys(g) )
    return mom_grad
end

"""
Loads MNIST train/test data
	- normalizes features by mapping pixels to (0,1)
"""
function load_mnist()
	train_set = CSV.File("../data/mnist/mnist_train.csv", header=false)
	test_set = CSV.File("../data/mnist/mnist_test.csv", header=false)

	train_x = map(i -> train_set[i][2:end] ./ 255, 1:length(train_set)) # normalize pixels to (0,1)
	test_x = map(i -> test_set[i][2:end]./ 255, 1:length(test_set)) # normalize pixels to (0,1)
	train_y = map(i -> train_set[i][1], 1:length(train_set))
	test_y = map(i -> test_set[i][1], 1:length(test_set))
	train_y = one_hot(train_y) # one hot labels
	test_y = one_hot(test_y) # one hot labels

	return train_x, train_y, test_x, test_y
end

"""
Returns test error for MNIST
"""
function mnist_test_error(network, features, labels, ntwk_type :: String)
	"""
	Compute test error over 10K MNIST samples
	"""
	#Verify prediction and true label are same size
	network.results != size(labels, 2) && @error "Prediction and true label are different dimensions."

	N = length(features)
	num_correct = zeros(N)
	x0 = ntwk_type == "lstm" ? 2*network.hid_dim : network.hid_dim

	for i=1:N
		U, label = features[i], labels[i,:]
		# append initial hidden state (zero vector)
		U = vcat(U, zeros(x0))
		pred = Network.evaluate(network, U)[end-network.results+1:end]
		Y = findmax(pred)[2]
		num_correct[i] = Y == findmax(label)[2] ? 1 : 0 ;
   end
   return 1.0 - (sum(num_correct) / N)
end

end # end module
