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

	init_F, init_error = Helpers.mnist_test_error(ntwk, test_x, test_y, loss, ntwk_type)
	printstyled("Initial Test Error:\t$(init_error)\n\n", color=:cyan)
	printstyled("Initial Test F:\t$(init_F)\n\n", color=:cyan)

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

		for j=1:N#60,000 training samples

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
		#file_name = string("../analysis/mnist/$(ntwk_type)_$(method)_1e2_$(i).jld2")
		#save(file_name, "ntwk", ntwk)

	end
	return [epoch_info]
end


# TRAINING PARAMETERS/CONSTANTS
NTWK_TYPE = "lstm"
METHOD = "adjoint"
#G_FACTOR = "1e2"
SEED = 1 #parse(Int64, ENV["seed"])
HID_DIM = 20
OUTPUT_DIM = 10
INP_DIM = 784 # flattened feature image (28 × 28)
FC_DIM = 0 # dimension of fully connect layer (optional)
SEQ_LENGTH = 28 # number of time steps in sequence
η = 0.001 # learning rate default for adam (sgd = 0.01)
BATCH_SIZE = 32
NUM_EPOCHS = 5
LSMOD = loss.softmaxCrossEntropy # loss
PNMOD = penalty.var_phi # adjoint control function
OPT = "adam"

# intiailize network, partition and verify
NTWK, PARTITION, SYNCFLAG = NTWK_TYPE == "rnn" ?
		RNN.specifyRNN(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
		NTWK_TYPE == "lstm" ?
				LSTM.specifyLSTM(INP_DIM+2*HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
				GRU.specifyGRU(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED)


# load ntwk trained parameters
KERAS_WEIGHTS = npzread("mnist-params/keras-lstm-mnist-adam.npz")
W_f = KERAS_WEIGHTS["W_f"]
W_i = KERAS_WEIGHTS["W_i"]
W_o = KERAS_WEIGHTS["W_o"]
W_c = KERAS_WEIGHTS["W_c"]
W_out = KERAS_WEIGHTS["W_out"]

# seed network with keras pretrained parameters
#NTWK_NEW = Helpers.set_lstm_params(NTWK, PARTITION, W_f, W_i, W_c, W_o, W_out)
#printstyled("Network Loading Complete.\n", color=:green)

# load ntwk gradients
KERAS_GRADS = npzread("mnist-params/keras-lstm-mnist-adam-grad.npz")
dW_f = KERAS_GRADS["dW_f"]
dW_i = KERAS_GRADS["dW_i"]
dW_o = KERAS_GRADS["dW_o"]
dW_c = KERAS_GRADS["dW_c"]
dW_out = KERAS_GRADS["dW_out"]






# load and transform input-output pair
NTWK_NEW = Helpers.set_lstm_params(NTWK, PARTITION, W_f, W_i, W_c, W_o, W_out)
VERIFY_PAIR = npzread("mnist-params/keras-lstm-mnist-test-pair.npz")


label = VERIFY_PAIR["label"]
U = vcat(collect(Iterators.flatten(VERIFY_PAIR["U"][1,:,:]')), zeros(2*NTWK_NEW.hid_dim))
label = convert.(Float64, map(i -> i==5 ? 1 : 0, 0:9))

# Evaluate input/output pair with julia implementation
# evaluate gradient
dP = paramGrad(LSMOD, NTWK_NEW, U, label, SYNCFLAG, PARTITION)

# verify gradients for first neuron
julia_grad_rec = dP[3241] # note: 3241 is max index in partition containing neuron 1
keras_grad_rec = dW_f[:,1]

# verify gradients for output neurons
julia_grad_out = dP[3361] # note: 3361 is first index in partition containing output (sigmoid neuron)
keras_grad_out = dW_out[:,1]

ϵ = 1e-6
printstyled("Testing Julia and Keras Gradients:\n", color=:yellow)

printstyled("\tRecurrent Gradient:\t", color=:cyan)
norm(keras_grad_rec - julia_grad_rec) < ϵ ?
	printstyled("\tPASSED\n", color=:green) :
	printstyled("\tFAILED\n", color=:red)

printstyled("\tOutput Gradient:\t", color=:cyan)
norm(keras_grad_out - julia_grad_out) < ϵ ?
	printstyled("\tPASSED\n", color=:green) :
	printstyled("\tFAILED\n", color=:red)


# train ntwk
printstyled("Starting Training on Sequential MNIST Task:\t$(METHOD) -- $(OPT)\n\n", color=:yellow)
#TRAINING_OUTPUT = train(NTWK_NEW, LSMOD, PNMOD, η, NUM_EPOCHS, METHOD, SYNCFLAG, PARTITION, NTWK_TYPE, BATCH_SIZE, OPT)
