# Sequential MNIST Training
# Author: Liam Johnston

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/rnn.jl")
#include("../src/lstm.jl")
#include("../src/gru.jl")

using FileIO, NPZ, LinearAlgebra, CSV, Random, JLD2, SparseArrays, MLDatasets, Random, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN#, Main.LSTM, Main.GRU


"""
MNIST Training Function
	- Returns dictionary of vectors storing epoch training information
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String, batch_size :: Int64, optimizer :: String)

	# load MNIST data
	test_x,  test_y  = MNIST.testdata()
	# reshape features and one-hot encode labels
	test_x = map(i -> collect(Iterators.flatten(test_x[:,:,i])), 1:size(test_x,3))
	test_y = Helpers.one_hot(test_y)

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)
	x0 = ntwk_type == "lstm" ? 2*ntwk.hid_dim : ntwk.hid_dim

	#init_F, init_error = Helpers.mnist_test_error(ntwk, test_x, test_y, loss, ntwk_type)
	#printstyled("Initial Test Error:\t$(init_error)\n", color=:cyan)
	#printstyled("Initial Test F:\t\t$(init_F)\n\n", color=:cyan)


	for i=1:num_epochs
		# load and permute training batches
		train_x, train_y = Helpers.batch("mnist", 32, i)
		F = 0.0
		for j=1:length(train_y) #iterate through batches
			batch_x, batch_y = train_x[j], train_y[j]
			b_grads, b_F = Helpers.batch_gradient(ntwk, batch_x, batch_y, loss, syncFlag, partition)
			F += b_F
			# update parameters
			optimizer == "adam" ?
				adam_update!(ntwk, b_grads, η, syncFlag, partition, m, v, i) :
				update!(ntwk, b_grads, η, syncFlag, partition)
			if j == 1 || j % 50 == 0 || j ==length(train_y)
				printstyled("\r\tBatch: $(j)/$(length(train_y)) completed.", color=:light_cyan, bold=:true)
			end
		end

		# classification error over test set
		test_F, test_error = Helpers.mnist_test_error(ntwk, test_x, test_y, loss, ntwk_type)

		printstyled("\n\n Epoch: $(i)\n", color=:light_cyan, bold=:true)
		printstyled("\tError: $(test_error)\n", color=:light_magenta)
		printstyled("\tTrain F: $(F)\n", color=:light_magenta)
		printstyled("\tTest F: $(test_F)\n", color=:light_magenta)

	end # end epoch

	return
end


# TRAINING PARAMETERS/CONSTANTS
NTWK_TYPE = "rnn"
METHOD = "adjoint"
#G_FACTOR = "1e2"
SEED = 1 #parse(Int64, ENV["seed"])
HID_DIM = 20
OUTPUT_DIM = 10
INP_DIM = 784 # flattened feature image (28 × 28)
FC_DIM = 0 # dimension of fully connect layer (optional)
SEQ_LENGTH = 28 # number of time steps in sequence
η = 0.001 # learning rate default for adam (sgd = 0.01)
BATCH_SIZE = 32#500
NUM_EPOCHS = 50
LSMOD = loss.softmaxCrossEntropy # loss
#LSMOD2 = loss.crossEntropy
PNMOD = penalty.var_phi # adjoint control function
OPT = "adam"

# intiailize network, partition and verify
NTWK, PARTITION, SYNCFLAG = NTWK_TYPE == "rnn" ?
		RNN.specifyRNN(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
		NTWK_TYPE == "lstm" ?
				LSTM.specifyLSTM(INP_DIM+2*HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
				GRU.specifyGRU(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED)

KERAS_WEIGHTS = npzread("mnist-params/keras-rnn-mnist-adam-one.npz")
W_REC_K = vcat(KERAS_WEIGHTS["w_inp"],KERAS_WEIGHTS["w_rec"])
W_REC_BIAS_K = KERAS_WEIGHTS["w_rec_bias"]
W_OUT_K = KERAS_WEIGHTS["w_out"]
W_OUT_BIAS_K = KERAS_WEIGHTS["w_out_bias"]

# seed network with keras pretrained parameters
NTWK_NEW = Helpers.set_rnn_params(NTWK, PARTITION, W_REC_K, W_REC_BIAS_K, W_OUT_K, W_OUT_BIAS_K)


# load ntwk gradients
KERAS_GRADS = npzread("mnist-params/keras-rnn-mnist-adam-grad-one.npz")
dW_REC_K = vcat(KERAS_GRADS["dw_inp"],KERAS_GRADS["dw_rec"])
dW_REC_BIAS_K = KERAS_GRADS["dw_rec_bias"]
dW_OUT_K = KERAS_GRADS["dw_out"]
dW_OUT_BIAS_K = KERAS_GRADS["dw_out_bias"]

# load and transform input-output pair
VERIFY_PAIR = npzread("mnist-params/keras-rnn-mnist-test-pair-one.npz")
label = VERIFY_PAIR["label"]
U = vcat(collect(Iterators.flatten(VERIFY_PAIR["U"][1,:,:]')), zeros(NTWK_NEW.hid_dim))
label = convert.(Float64, map(i -> i==5 ? 1 : 0, 0:9))

# evaluate gradient
dP = paramGrad(LSMOD, NTWK_NEW, U, label, SYNCFLAG, PARTITION)

# verify gradients for first neuron
julia_grad_rec = dP[541] # note: 541 is max index in partition containing neuron 1
keras_grad_rec = vcat(dW_REC_BIAS_K[1], dW_REC_K[:,1])

# verify gradients for output neurons
julia_grad_out = dP[561] # note: 561 is first index in partition containing output (sigmoid neuron)
keras_grad_out = vcat(dW_OUT_BIAS_K[1], dW_OUT_K[:,1])

ϵ = 1e-5
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
printstyled("Starting Training on Batched Sequential MNIST Task:\n\tTraining Method:\t$(METHOD)\n\tOptimizer:\t\t$(OPT)\n\n", color=:yellow)

TRAINING_OUTPUT = train(NTWK_NEW, LSMOD, PNMOD, η, NUM_EPOCHS, METHOD, SYNCFLAG, PARTITION, NTWK_TYPE, BATCH_SIZE, OPT)
