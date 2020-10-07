# Verifies tensorflow parameters on MNIST dataset
# Author: Liam Johnston

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/rnn.jl")
include("../src/lstm.jl")
include("../src/gru.jl")

using FileIO, NPZ, LinearAlgebra, MLDatasets, CSV, Random, JLD2, SparseArrays, MLDatasets, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU


"""
MNIST Training Function
	- Returns dictionary of vectors storing epoch training information
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String, batch_size :: Int64, optimizer :: String)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs)
						)

	# load MNIST data
	train_x, train_y = MNIST.traindata()
	test_x,  test_y  = MNIST.testdata()
	# reshape features and one-hot encode labels
	train_x = map(i -> collect(Iterators.flatten(train_x[:,:,i])), 1:size(train_x,3))
	test_x = map(i -> collect(Iterators.flatten(test_x[:,:,i])), 1:size(test_x,3))
	train_y, test_y = Helpers.one_hot(train_y), Helpers.one_hot(test_y)

	N = length(train_x)

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)
	x0 = ntwk_type == "lstm" ? 2*ntwk.hid_dim : ntwk.hid_dim

	init_error = Helpers.mnist_test_error(ntwk, test_x, test_y, ntwk_type)
	printstyled("Initial Error:\t$(init_error)\n\n", color=:cyan)

	for i=1:num_epochs

		# time sgd update
		update_time = 0.0

		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))

		# shuffle training data
		Random.seed!(i)
		perm_i = shuffle(1:length(train_x))
		train_x_shuffle = train_x[perm_i]
		train_y_shuffle = train_y[perm_i,:]

		# count batch examples
		ct = 0
		batch_grad = []

		for j=1:5000#60,000 training samples

			U, label = vcat(train_x_shuffle[j], zeros(x0)), train_y_shuffle[j,:]
			X = Network.evaluate(ntwk, U)
			Λ = Network.adjoint(ntwk, X, label, loss)
			F += loss.evaluate(label, X, ntwk.results)
			G += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type)

			# append batch gradient
			method == "adjoint" ?
				append!(batch_grad, [paramGrad(loss, ntwk, U, label, syncFlag, partition)]) : # adjoint update
				append!(batch_grad, [paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type)]) # coadjoint update
			ct += 1

			if ct == batch_size
				# compute batch gradient
				update_grad = Helpers.batchGrad(batch_grad)

				# update ntwk params
				update_time = @elapsed begin
					optimizer == "adam" ?
						adam_update!(ntwk, update_grad, η, syncFlag, partition, m, v, i) :
						update!(ntwk, update_grad, η, syncFlag, partition)
				end

				# stop training if F and/or G are numerically unstable (inf or NaN produced)
				Helpers.verifyNumericalStability(F,G) == true ? continue : return [epoch_info]
				# reset batch_grad and ct
				batch_grad = []
				ct = 0
			end
		end

		λ_ind = ntwk_type == "rnn" ?
				RNN.rnnXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length) :
				ntwk_type == "lstm" ?
						LSTM.lstmXind(ntwk.features-2*ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length) :
						GRU.gruXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length)
		#if ntwk_type == "rnn"
		#	λ_ind = RNN.rnnXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length)
		#else
		#	λ_ind = ntwk_type == "lstm" ?
		#			LSTM.lstmXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length) :
		#			GRU.gruXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length)
		#end


		# classification error over test set
		error = Helpers.mnist_test_error(ntwk, test_x, test_y, ntwk_type)
		epoch_info["F"][i] = F / N
		epoch_info["G"][i] = G / N
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = update_time
		epoch_info["Λ_init"][i] = norm(Λ[λ_ind[1:ntwk.hid_dim]] ./ N)
		epoch_info["Λ_out"][i] = norm(Λ[λ_ind[end-ntwk.hid_dim+1:end]] ./ N)
		epoch_info["Error"][i] = error

		# WRITE EPOCH INFO TO FILE
		#epoch_file = "training-info/$(ntwk_type)/$(ntwk_type)_$(method)_$(seed).dat"
	    #out_file = open(epoch_file, "a")
			printstyled("\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
			printstyled("\tError: $(error)\n", color=:light_magenta)
			printstyled("\tObjective (F+G): $(F+G)\n", color=:light_magenta)
			printstyled("\tF: $(F/N)\n", color=:light_magenta)
			printstyled("\tG: $(G/N)\n", color=:light_magenta)
			printstyled("\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
			printstyled("\t||λΤ||: $(epoch_info["Λ_out"][i])\n", color=:light_magenta)
			printstyled("\tTime: $(update_time)\n\n", color=:light_magenta)
		#close(f)
		# save intermediate ntwk after epoch i
		#file_name = string("ntwks/$(ntwk_type)/$(ntwk_type)_$(method)_$(seed)_$(i).jld2")
		#save(file_name, "ntwk", ntwk)

	end
	return [epoch_info]
end


# TRAINING PARAMETERS/CONSTANTS
NTWK_TYPE = "rnn"
METHOD = "coadjoint"
SEED = 1 #parse(Int64, ENV["seed"])
HID_DIM = 20
OUTPUT_DIM = 10
INP_DIM = 784 # flattened feature image (28 × 28)
FC_DIM = 0 # dimension of fully connect layer (optional)
SEQ_LENGTH = 28 # number of time steps in sequence
η = 0.01 # learning rate
BATCH_SIZE = 125
NUM_EPOCHS = 50
LSMOD = loss.softmaxCrossEntropy # loss
PNMOD = penalty.var_phi # adjoint control function
OPT = "sgd"

# intiailize network, partition and verify
NTWK, PARTITION, SYNCFLAG = NTWK_TYPE == "rnn" ?
		RNN.specifyRNN(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
		NTWK_TYPE == "lstm" ?
				LSTM.specifyLSTM(INP_DIM+2*HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
				GRU.specifyGRU(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED)

# load ntwk trained parameters
W_REC = npzread("tf-rnn-vars/w-rec.npz")["name1"]
W_REC_BIAS = npzread("tf-rnn-vars/w-rec-bias.npz")["name1"]
W_OUT = npzread("tf-rnn-vars/w-out.npz")["name1"]
W_OUT_BIAS = npzread("tf-rnn-vars/w-out-bias.npz")["name1"]

NTWK_NEW = Helpers.set_rnn_params(NTWK, PARTITION, W_REC, W_REC_BIAS, W_OUT, W_OUT_BIAS)

#test_feats = npzread("tf-rnn-vars/tf_test_data.npz")["name1"] # (128,28,28)
#test_labels = npzread("tf-rnn-vars/tf_test_label.npz")["name1"] # (128, 10)
#test_feats = map(i -> test_feats[i,:,:], 1:size(test_feats,1))

#test_f = map(i -> collect(Iterators.flatten(test_feats[i]')), 1:size(test_feats,1))
# train ntwk
printstyled("Starting Training:\t$(METHOD) -- $(OPT)\n\n", color=:yellow)
TRAINING_OUTPUT = train(NTWK_NEW, LSMOD, PNMOD, η, NUM_EPOCHS, METHOD, SYNCFLAG, PARTITION, NTWK_TYPE, BATCH_SIZE, OPT)
