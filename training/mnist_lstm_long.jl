seed = parse(Int64, ENV["seed"])

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/lstm.jl")


using LinearAlgebra, Random, JLD, SparseArrays, MLDatasets, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.LSTM

"""
Implements MNIST RNN Training on Partial Dataset (corresponding to label dim)
"""

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs))

	train_x, train_y = MNIST.traindata()
	test_x,  test_y  = MNIST.testdata()
	train_y = Helpers.one_hot(train_y) .+ ones(size(train_y,1)) * 0.001
	test_y = Helpers.one_hot(test_y)

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)

	for i=1:num_epochs

		# time sgd update
		update_time = 0.0

		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))

		# shuffle training data
		Random.seed!(i)
		perm_i = shuffle(1:size(train_x,3))
		train_x_shuffle = train_x[:,:,perm_i]
		train_y_shuffle = train_y[perm_i,:]

		for j=1:5#size(train_x, 3)
			println(j)

			batch_grad = []
			batch_size = 4

			for i=1:batch_size

				U, label = collect(Iterators.flatten(train_x_shuffle[:,:,j])), train_y_shuffle[j,:]
				U = vcat(U, zeros(2*ntwk.hid_dim))
				X = Network.evaluate(ntwk, U)
				Λ = Network.adjoint(ntwk, X, label, loss)
				F += loss.evaluate(label, X, ntwk.results) / batch_size
				G += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type) / batch_size

				# append batch gradient
				method == "adjoint" ?
					append!(batch_grad, [paramGrad(loss, ntwk, U, label, syncFlag, partition)]) : # adjoint update
					append!(batch_grad, [paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type)]) # coadjoint update
			end

			# compute batch gradient
			update_grad = Helpers.batchGrad(batch_grad)

			# update ntwk params
			update_time = @elapsed begin
				adam_update!(ntwk, update_grad, η, syncFlag, partition, m, v, i)
			end

			# stop training if F and/or G are numerically unstable (inf or NaN produced)
			Helpers.verifyNumericalStability(F,G) == true ? continue : return [epoch_info]
		end

		# classification error over test set
		error = LSTM.classification_error(ntwk, test_x, test_y)

		epoch_info["F"][i] = F
		epoch_info["G"][i] = G
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = update_time
		λ_ind = LSTM.lstmXind(ntwk.features-2*ntwk.hid_dim, ntwk.hid_dim, ntwk.seq_length)
		epoch_info["Λ_init"][i] = norm(Λ[λ_ind[1:hid_dim]] ./ size(train_x,3))
		epoch_info["Λ_out"][i] = norm(Λ[λ_ind[end-hid_dim+1:end]] ./ size(train_x,3))
		epoch_info["Error"][i] = error

		# WRITE EPOCH INFO TO FILE
		epoch_file = "training-info/lstm/lstm_$(method)_$(seed)_long.dat"
	    f = open(epoch_file, "a")
			printstyled(f, "\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
			printstyled(f, "\tError: $(error)\n", color=:light_magenta)
			printstyled(f, "\tObjective (F+G): $(F+G)\n", color=:light_magenta)
			printstyled(f, "\tF: $(F/size(train_x, 3))\n", color=:light_magenta)
			printstyled(f, "\tG: $(G/size(train_x, 3))\n", color=:light_magenta)
			printstyled(f, "\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
			printstyled(f, "\t||λΤ||: $(epoch_info["Λ_out"][i])\n", color=:light_magenta)
			printstyled(f, "\tTime: $(update_time)\n\n", color=:light_magenta)
		close(f)
		# save intermediate ntwk after epoch i
		file_name = string("ntwks/lstm/lstm_$(method)_$(seed)_$(i)_long.jld")
		save(file_name, "ntwk", ntwk)


	end
	return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
# network structure
ntwk_type = "lstm"
method = "adjoint"
hid_dim = 50
output_dim = 10
inp_dim = 784 + 2*hid_dim # corresponds to ntwk.features (input dim + inital hidden state dim)
fc_dim = 0 # dimension of fully connect layer post recursive layers
seq_length = 784 # number of time points
η = 0.01 # learning rate
num_epochs = 100
lsmod = loss.crossEntropy
pnmod = penalty.var_phi_lstm
ntwk, partition, syncFlag = LSTM.specifyLSTM(inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)

# write network training parameters to file
epoch_file = "training-info/lstm/lstm_$(method)_$(seed)_long.dat"


f = open(epoch_file, "a")
	printstyled(f, "**************\tTraining Method: LSTM $(method)\t**************\n\n", color = :yellow, bold = :true)
    printstyled(f, "\tSeed: $(seed)\n", color = :light_cyan)
    printstyled(f, "\tLearning Rate: $(η)\n", color = :light_cyan)
    printstyled(f, "\tLoss (F): Cross Entropy (Softmax)\n", color = :light_cyan)
	printstyled(f, "\tPenalty (G): Var + ϕ\n", color = :light_cyan)
	printstyled(f, "\tNetwork Architecture:\n", color=:light_cyan)
	printstyled(f, "\t\tInput Dim: $(inp_dim) (784 input + $(hid_dim) init state)\n", color = :light_cyan)
	printstyled(f, "\t\tHidden Dim: $(hid_dim)\n", color = :light_cyan)
	printstyled(f, "\t\tFully Connect Dim: $(fc_dim)\n", color = :light_cyan)
	printstyled(f, "\t\tSoftmax Dim: $(output_dim)\n\n", color = :light_cyan)
	printstyled("***************************************************************\n\n", color = :yellow, bold = :true)
close(f)

# train ntwk
training_output = train(ntwk, lsmod, pnmod, η, num_epochs, method, syncFlag, partition, ntwk_type)

# save trained network
file_name = string("ntwks/lstm/lstm_$(method)_trained_long.jld")
save(file_name, "ntwk", ntwk)
