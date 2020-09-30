seed = 2 #parse(Int64, ENV["seed"])

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/rnn.jl")


using LinearAlgebra, Random, JLD, SparseArrays, NPZ, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN

"""
Implements MNIST RNN Training on Partial Dataset (corresponding to label dim)
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs))

	train_x = npzread("../data/urban-sounds/train_x.npy")
	train_y = npzread("../data/urban-sounds/train_y.npy")
	test_x = npzread("../data/urban-sounds/test_x.npy")
	test_y = npzread("../data/urban-sounds/test_y.npy")

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)
	batch_size = 4
	num_updates = size(train_x,1) / batch_size

	for i=1:num_epochs
		# Store Epoch Variables
		update_time = 0.0
		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))

		# shuffle training data
		Random.seed!(i)
		perm_i = shuffle(1:size(train_x,1))
		train_x_shuffle = train_x[perm_i,:,:]
		train_y_shuffle = train_y[perm_i,:]

		# count batch examples
		ct = 0

		for j=1:size(train_x, 1)
			ct += 1
			batch_grad = []

			U, label = collect(Iterators.flatten(train_x_shuffle[j,:,:])), train_y_shuffle[j,:]
			U = vcat(U, zeros(ntwk.hid_dim))
			X = Network.evaluate(ntwk, U)
			Λ = Network.adjoint(ntwk, X, label, loss)
			F += loss.evaluate(label, X, ntwk.results) / batch_size
			G += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type) / batch_size

			# append batch gradient
			method == "adjoint" ?
				append!(batch_grad, [paramGrad(loss, ntwk, U, label, syncFlag, partition)]) : # adjoint update
				append!(batch_grad, [paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type)]) # coadjoint update

			if ct == batch_size
				# compute batch gradient
				update_grad = Helpers.batchGrad(batch_grad)

				# update ntwk params
				update_time = @elapsed begin
					adam_update!(ntwk, update_grad, η, syncFlag, partition, m, v, i)
				end

				# Verify Numerical Stability
				Helpers.verifyNumericalStability(F,G) == true ? continue : return [epoch_info]

				# reset batch_grad and ct
				batch_grad = []
				ct = 0
			end
		end

		# classification error over test set
		error = Helpers.urban_classification_error(ntwk, test_x, test_y)

		epoch_info["F"][i] = F
		epoch_info["G"][i] = G
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = update_time
		λ_ind = RNN.rnnXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length)
		epoch_info["Λ_init"][i] = norm(Λ[λ_ind[1:hid_dim]] ./ num_updates)
		epoch_info["Λ_out"][i] = norm(Λ[λ_ind[end-hid_dim+1:end]] ./ num_updates)
		#epoch_info["Error"][i] = error

		# WRITE EPOCH INFO TO FILE
		#epoch_file = "training-info/rnn/rnn_$(method)_$(seed).dat"
	    #out_file = open(epoch_file, "a")
			printstyled("\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
			printstyled("\tError: $(error)\n", color=:light_magenta)
			printstyled("\tObjective (F+G): $((F+G)/num_updates)\n", color=:light_magenta)
			printstyled("\tF: $(F/num_updates)\n", color=:light_magenta)
			printstyled("\tG: $(G/num_updates)\n", color=:light_magenta)
			printstyled("\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
			printstyled("\t||λΤ||: $(epoch_info["Λ_out"][i])\n", color=:light_magenta)
			printstyled("\tTime: $(update_time)\n\n", color=:light_magenta)
		#close(f)
		# save intermediate ntwk after epoch i
		#file_name = string("ntwks/rnn/rnn_$(method)_$(seed)_$(i).jld")
		#save(file_name, "ntwk", ntwk)

	end
	return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
# network structure
ntwk_type = "rnn"
method = "adjoint"
hid_dim = 5
output_dim = 10
inp_dim = 6960 + hid_dim # corresponds to ntwk.features (input dim + inital hidden state dim)
fc_dim = 0 # dimension of fully connect layer post recursive layers
seq_length = 174 # number of time points
ntwk, partition, syncFlag = RNN.specifyRNN(inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)
η = 0.01 # learning rate
num_epochs = 100
lsmod = loss.crossEntropy
pnmod = penalty.var_phi

# write network training parameters to file
#epoch_file = "training-info/rnn/rnn_$(method)_$(seed).dat"


#output_file = open(epoch_file, "a")
	printstyled("**************\tTraining Method: RNN $(method)\t**************\n\n", color = :yellow, bold = :true)
    printstyled("\tSeed: $(seed)\n", color = :light_cyan)
    printstyled("\tLearning Rate: $(η)\n", color = :light_cyan)
    printstyled("\tLoss (F): Cross Entropy (Softmax)\n", color = :light_cyan)
	printstyled("\tPenalty (G): Var + ϕ\n", color = :light_cyan)
	printstyled("\tNetwork Architecture:\n", color=:light_cyan)
	printstyled("\t\tInput Dim: $(inp_dim)\n", color = :light_cyan)
	printstyled("\t\tHidden Dim: $(hid_dim)\n", color = :light_cyan)
	printstyled("\t\tFully Connect Dim: $(fc_dim)\n", color = :light_cyan)
	printstyled("\t\tSoftmax Dim: $(output_dim)\n\n", color = :light_cyan)
	printstyled("***************************************************************\n\n", color = :yellow, bold = :true)
#close(output_file)

# train ntwk
training_output = train(ntwk, lsmod, pnmod, η, num_epochs, method, syncFlag, partition, ntwk_type)

# save trained network
#file_name = string("ntwks/rnn/rnn_$(method)_trained.jld")
#save(file_name, "ntwk", ntwk)
