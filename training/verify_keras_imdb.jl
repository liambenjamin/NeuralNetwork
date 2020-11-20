# Verifies Tensorflow-Keras parameters on MNIST dataset
# Author: Liam Johnston

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/rnn.jl")
include("../src/lstm.jl")
include("../src/gru.jl")

using FileIO, NPZ, LinearAlgebra, CSV, Random, JLD2, SparseArrays, MLDatasets, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU


"""
MNIST Training Function
	- Returns dictionary of vectors storing epoch training information
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String, batch_size :: Int64, optimizer :: String, embedding :: Matrix)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs)
						)

	# Load IMDB data
	train_x = npzread("imdb-params/keras-rnn-imdb-train-features.npz")["X"]
	test_x = npzread("imdb-params/keras-rnn-imdb-test-features.npz")["X"]
	train_y = npzread("imdb-params/keras-rnn-imdb-labels.npz")["y_train"]
	test_y = npzread("imdb-params/keras-rnn-imdb-labels.npz")["y_test"]

	N = size(train_x,1)

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)
	x0 = ntwk_type == "lstm" ? 2*ntwk.hid_dim : ntwk.hid_dim


	init_error = Helpers.imdb_test_error(ntwk, test_x, test_y, embedding, ntwk_type)
	printstyled("Initial Error:\t$(init_error)\n\n", color=:cyan)

	for i=1:num_epochs

		# time sgd update
		update_time = 0.0

		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))

		# shuffle training data
		Random.seed!(i)
		perm_i = shuffle(1:size(train_x,1))
		train_x_shuffle = train_x[perm_i,:]
		train_y_shuffle = train_y[perm_i]

		# count batch examples
		ct = 0
		batch_grad = []

		for j=1:N#60,000 training samples

			U, label = Helpers.embed(features[j,:], embedding), labels[j]
			U = vcat(collect(Iterators.flatten(U')), zeros(ntwk.hid_dim))
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

		# classification error over test set
		error = Helpers.imdb_test_error(ntwk, test_x, test_y, embedding, ntwk_type)
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
		#file_name = string("../analysis/trained-ntwks/$(ntwk_type)/$(ntwk_type)_$(method)_$(i).jld2")
		#save(file_name, "ntwk", ntwk)

	end
	return [epoch_info]
end


# TRAINING PARAMETERS/CONSTANTS
NTWK_TYPE = "rnn"
METHOD = "adjoint"
SEED = 1 #parse(Int64, ENV["seed"])
HID_DIM = 40
OUTPUT_DIM = 1
INP_DIM = 20000 # (200 × 100) --> 500 words embedded in R^(100)
FC_DIM = 0 # dimension of fully connect layer (optional)
SEQ_LENGTH = 200 # number of time steps in sequence
η = 0.001 # learning rate default for adam (sgd = 0.01)
BATCH_SIZE = 32
NUM_EPOCHS = 5
LSMOD = loss.binary_crossEntropy # loss
PNMOD = penalty.var_phi # adjoint control function
OPT = "adam"
E = npzread("imdb-params/keras-rnn-imdb-embedding-matrix.npz")["E"]

# intiailize network, partition and verify
NTWK, PARTITION, SYNCFLAG = NTWK_TYPE == "rnn" ?
		RNN.specifyRNN(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
		NTWK_TYPE == "lstm" ?
				LSTM.specifyLSTM(INP_DIM+2*HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
				GRU.specifyGRU(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED)

# load ntwk trained parameters
KERAS_WEIGHTS = npzread("imdb-params/keras-rnn-imdb-adam.npz")
W_REC_K = vcat(KERAS_WEIGHTS["w_inp"],KERAS_WEIGHTS["w_rec"])
W_REC_BIAS_K = KERAS_WEIGHTS["w_rec_bias"]
W_OUT_K = KERAS_WEIGHTS["w_out"]
W_OUT_BIAS_K = KERAS_WEIGHTS["w_out_bias"]

# seed network with keras pretrained parameters
NTWK_NEW = Helpers.set_rnn_params(NTWK, PARTITION, W_REC_K, W_REC_BIAS_K, W_OUT_K, W_OUT_BIAS_K)

printstyled("Testing Julia and Keras Parameters:\n", color=:yellow)
ϵ = 1e-6
neur1_params_k = npzread("imdb-params/keras-rnn-imdb-neur1-params.npz")["neur1_params"]
neur1_params = NTWK_NEW.neurons[1].β
printstyled("\tTesting Neuron 1 Parameter Matching:\t", color=:cyan)
norm(neur1_params_k - neur1_params) < ϵ ?
	printstyled("\tPASSED\n", color=:green) :
	printstyled("\tFAILED\n", color=:red)

printstyled("Testing Julia and Keras Gradients:\n", color=:yellow)

# load ntwk gradients
KERAS_GRADS = npzread("imdb-params/keras-rnn-imdb-adam-grad.npz")
dW_REC_K = vcat(KERAS_GRADS["dw_inp"],KERAS_GRADS["dw_rec"])
dW_REC_BIAS_K = KERAS_GRADS["dw_rec_bias"]
dW_OUT_K = KERAS_GRADS["dw_out"]
dW_OUT_BIAS_K = KERAS_GRADS["dw_out_bias"]

test_x = npzread("imdb-params/keras-rnn-imdb-test-features.npz")["X"]
test_y = convert.(Float64, npzread("imdb-params/keras-rnn-imdb-labels.npz")["y_test"])
U = test_x[1,:]
label = [test_y[1]]
U = Helpers.embed(U,E)
U = vcat(collect(Iterators.flatten(U')), zeros(NTWK_NEW.hid_dim)) #flatten(200x100)+40
dP = paramGrad(LSMOD, NTWK_NEW, U, label, SYNCFLAG, PARTITION)

# verify gradients for first neuron
julia_grad_rec = dP[7961] # note: 7961 is max index in partition containing neuron 1
dP_neur1 = npzread("imdb-params/keras-rnn-imdb-neur1-grad.npz")["neur1_grad"]

# verify gradients for output neurons
julia_grad_out = dP[8001] # note: 8001 is first index in partition containing output (sigmoid neuron)
dP_neur_out = npzread("imdb-params/keras-rnn-imdb-neur-out-grad.npz")["neur_out_grad"]

printstyled("\tRecurrent Gradient:\t", color=:cyan)
norm(dP_neur1 - julia_grad_rec) < ϵ ?
	printstyled("\tPASSED\n", color=:green) :
	printstyled("\tFAILED\n", color=:red)

printstyled("\tOutput Gradient:\t", color=:cyan)
norm(dP_neur_out - julia_grad_out) < ϵ ?
	printstyled("\tPASSED\n", color=:green) :
	printstyled("\tFAILED\n", color=:red)


# train ntwk
#printstyled("Starting Training:\t$(METHOD) -- $(OPT)\n\n", color=:yellow)
#TRAINING_OUTPUT = train(NTWK_NEW, LSMOD, PNMOD, η, NUM_EPOCHS, METHOD, SYNCFLAG, PARTITION, NTWK_TYPE, BATCH_SIZE, OPT, E)
