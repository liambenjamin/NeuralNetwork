seed = 20 #parse(Int64, ENV["seed"])

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")


using LinearAlgebra, Random, JLD, SparseArrays, MLDatasets, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers

"""
Implements MNIST RNN Training on Partial Dataset (corresponding to label dim)
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, seed :: Int64)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs))

	train_feat, train_label = MNIST.traindata()
	test_feat,  test_label  = MNIST.testdata()
	train_ind = findall(x -> x == 2 || x == 3, train_label)
	test_ind = findall(x -> x == 2 || x == 3, test_label)
	train_x = train_feat[:,:,train_ind]
	train_y = train_label[train_ind]
	test_x = test_feat[:,:,test_ind]
	test_y = test_label[test_ind]
	# one hot encoding of labels
	train_y = Helpers.one_hot(train_y) .+ ones(size(train_y,1)) * 0.001
	test_y = Helpers.one_hot(test_y)

	# learning rate
	l_rate = η
	for i=1:num_epochs

		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))

		# set epoch learning rate as function of initial learning rate

		l_rate = Helpers.adjust_lr(epoch_info["F"], epoch_info["G"], i, method) == false ? l_rate : 0.5 * l_rate

		# time sgd update
		update_time = 0.0

		# shuffle training data
		Random.seed!(seed + i)
		perm_i = shuffle(1:size(train_x,3))
		train_x_shuffle = train_x[:,:,perm_i]
		train_y_shuffle = train_y[perm_i,:]

		for j=1:size(train_x, 3)
			U, label = collect(Iterators.flatten(train_x_shuffle[:,:,j])), train_y_shuffle[j,:]
			U = vcat(U, zeros(ntwk.hid_dim))
			X = Network.evaluate(ntwk, U)
			Λ = Network.adjoint(ntwk, X, label, loss)

			F += loss.evaluate(label, X, ntwk.results)
			G += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length)

			# update ntwk params
			if method == "adjoint"
				update_time = @elapsed begin
					#update!(df, ntwk, U, label, l_rate) # no partition
					update!(loss, ntwk, U, label, l_rate, syncFlag, partition) # with partition
				end
			else
				update_time = @elapsed begin
					#update!(df, d2f, dg, ntwk, U, label, l_rate) # no partition
					update!(loss, penalty, ntwk, U, label, l_rate, syncFlag, partition) # with partition

				end
			end

			# stop training if F and/or G are numerically unstable (inf or NaN produced)
			Helpers.verifyNumericalStability(F,G) == true ? continue : return [epoch_info]
		end

		# classification error over test set
		error = Helpers.classification_error(ntwk, test_x, test_y)

		epoch_info["F"][i] = F
		epoch_info["G"][i] = G
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = update_time
		epoch_info["Λ_init"][i] = norm(Λ[ntwk.features-ntwk.hid_dim+1:ntwk.features] ./ size(train_x,3))
		epoch_info["Λ_out"][i] = norm(Λ[ntwk.features-ntwk.hid_dim+1+ntwk.hid_dim*ntwk.seq_length:ntwk.features+ntwk.hid_dim*ntwk.seq_length] ./ size(train_x,3))
		epoch_info["Error"][i] = error

		# WRITE EPOCH INFO TO FILE
		#epoch_file = "training_info/rnn_$(method)/sig_$(seed).dat"
	    #out_file = open(epoch_file, "a")
			printstyled("\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
			printstyled("\tError: $(error)\n", color=:light_magenta)
			printstyled("\tLearning Rate: $(l_rate)\n", color=:light_magenta)
			printstyled("\tObjective (F+G): $(F+G)\n", color=:light_magenta)
			printstyled("\tF: $(F)\n", color=:light_magenta)
			printstyled("\tG: $(G)\n", color=:light_magenta)
			printstyled("\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
			printstyled("\t||λΤ||: $(epoch_info["Λ_out"][i])\n", color=:light_magenta)
			printstyled("\tTime: $(update_time)\n\n", color=:light_magenta)
		#close(f)

	end
	return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
# network structure
method = "coadjoint"
hid_dim = 1
output_dim = 2
inp_dim = 784 + hid_dim # corresponds to ntwk.features (input dim + inital hidden state dim)
fc_dim = 10 # dimension of fully connect layer post recursive layers
seq_length = 28 # number of time points
ntwk, partition, syncFlag = Helpers.specifyStackedRNN(inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)
# TRAINING HYPER PARAMS
η = 0.0001 # learning rate
num_epochs = 500
lsmod = loss.crossEntropy
pnmod = penalty.var_phi


# write network training parameters to file
#epoch_file = "training_info/rnn_$(method)/sig_$(seed).dat"


#output_file = open(epoch_file, "a")
	printstyled("**************\tTraining Method: RNN $(method)\t**************\n\n", color = :yellow, bold = :true)
    printstyled("\tSeed: $(seed)\n", color = :light_cyan)
    printstyled("\tLearning Rate: $(η)	\n", color = :light_cyan)
    printstyled("\tLoss (F): Cross Entropy (Softmax)\n", color = :light_cyan)
	printstyled("\tPenalty (G): Var Phi\n", color = :light_cyan)
	printstyled("\tNetwork Architecture:\n", color=:light_cyan)
	printstyled("\t\tInput Dim: $(inp_dim) (784 input + $(hid_dim) init state)\n", color = :light_cyan)
	printstyled("\t\tHidden Dim: $(hid_dim)\n", color = :light_cyan)
	printstyled("\t\tFully Connect Dim: $(fc_dim)\n", color = :light_cyan)
	printstyled("\t\tSoftmax Dim: $(output_dim)\n\n", color = :light_cyan)
	printstyled("******************************************************************\n\n", color = :yellow, bold = :true)
#close(output_file)

training_output = train(ntwk, lsmod, pnmod, η, num_epochs, method, syncFlag, partition, seed)

# write experiment to ledger
#ledger = "Desktop/NN_Implementation/training/ledger.csv"
#dataset = "MNIST"
#label_set = "Labels: 2,3,4"
#loss_name = "Cross Entropy"
penalty_name = "Var Phi"
#Helpers.exp2file(ledger, dataset, label_set, seed, method, η, hid_dim, loss_name, penalty_name, training_output[1])

# save trained network
#file_name = string("$(penalty_name)_$(seed).jld")
#save(file_name, "ntwk", ntwk)
#ntwk_test = load("ntwk_test.jld", "ntwk")
