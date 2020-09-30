# MNIST Training Script
# Author: Liam Johnston

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/rnn.jl")
include("../src/lstm.jl")
include("../src/gru.jl")

using FileIO, LinearAlgebra, CSV, Random, JLD2, SparseArrays, MLDatasets, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU


"""
MNIST Training Function
	- Returns dictionary of vectors storing epoch training information
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String, batch_size :: Int64)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs)
						)

	# features (pixels) mapped to (0,1)
	# labels one-hot encoded
	train_x, train_y, test_x, test_y = Helpers.load_mnist()
	N = length(train_x)

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)
	x0 = ntwk_type == "lstm" ? 2*ntwk.hid_dim : ntwk.hid_dim

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

		for j=1:10#N#60,000 training samples

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
					adam_update!(ntwk, update_grad, η, syncFlag, partition, m, v, i)
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
NTWK_TYPE = "gru"
METHOD = "adjoint"
SEED = 1 #parse(Int64, ENV["seed"])
HID_DIM = 20
OUTPUT_DIM = 10
INP_DIM = 784 # flattened feature image (28 × 28)
FC_DIM = 0 # dimension of fully connect layer (optional)
SEQ_LENGTH = 28 # number of time steps in sequence
η = 0.1 # learning rate
BATCH_SIZE = 125
NUM_EPOCHS = 50
LSMOD = loss.softmaxCrossEntropy # loss
PNMOD = penalty.var_phi # adjoint control function

# intiailize network, partition and verify
NTWK, PARTITION, SYNCFLAG = NTWK_TYPE == "rnn" ?
		RNN.specifyRNN(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
		NTWK_TYPE == "lstm" ?
				LSTM.specifyLSTM(INP_DIM+2*HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED) :
				GRU.specifyGRU(INP_DIM+HID_DIM, OUTPUT_DIM, HID_DIM, FC_DIM, SEQ_LENGTH, SEED)


# write network training parameters to file
#epoch_file = "training-info/$(ntwk_type)/$(ntwk_type)_$(method)_$(seed).dat"


#output_file = open(epoch_file, "a")
	printstyled("**************\tTraining Method: $(NTWK_TYPE) $(METHOD)\t**************\n\n", color = :yellow, bold = :true)
    printstyled("\tSeed:\t$(SEED)\n", color = :light_cyan)
    printstyled("\tLearning Rate:\t$(η)\n", color = :light_cyan)
    printstyled("\tLoss (F):\tSoftmax Cross Entropy \n", color = :light_cyan)
	printstyled("\tPenalty (G):\tVar + ϕ\n", color = :light_cyan)
	printstyled("\tNetwork Architecture:\n", color=:light_cyan)
	printstyled("\t\t\tFeature Dim: $(INP_DIM)\n", color = :light_cyan)
	printstyled("\t\t\tHidden Dim: $(HID_DIM)\n", color = :light_cyan)
	printstyled("\t\t\tOutput Dim: $(OUTPUT_DIM)\n\n", color = :light_cyan)
	printstyled("***************************************************************\n\n", color = :yellow, bold = :true)
#close(output_file)

# train ntwk
TRAINING_OUTPUT = train(NTWK, LSMOD, PNMOD, η, NUM_EPOCHS, METHOD, SYNCFLAG, PARTITION, NTWK_TYPE, BATCH_SIZE)

# save trained network
#file_name = string("ntwks/$(NTWK_TYPE)/$(NTWK_TYPE)_$(METHOD)_$(SEED)_trained.jld")
#save(file_name, "ntwk", ntwk)
