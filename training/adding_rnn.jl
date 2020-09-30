include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/rnn.jl")


using FileIO, LinearAlgebra, Random, JLD2, SparseArrays, Colors, Main.Neurons, Main.Network, Main.Partition, Main.loss, Main.penalty, Main.Helpers, Main.RNN


"""
Implements RNN Training on Toy Experiment
"""
function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, seed :: Int64, batch_size :: Int64, ntwk_type :: String)

    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs),
					"Epoch" => zeros(num_epochs), "Time" => zeros(num_epochs),
					"Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
					"Λ_out" => zeros(num_epochs))

	m, v = Helpers.adam_init(ntwk, syncFlag, partition)

	for i=1:num_epochs

		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))

		# sgd update time
		update_time = 0.0

		n_train = 1000

		for j = 1:n_train
			batch_grad = []
			for k = 1:batch_size
				# U w/ appended initial state, x₀
				U, label = vcat(rand(10), zeros(ntwk.hid_dim)), zeros(ntwk.results)
				U[1:10] = map(i -> U[i] > 0.5 ? 1.0 : 0.0, 1:10)
				label[convert(Int64, (sum(U)+1))] = 1.0

				# forward states
				X = Network.evaluate(ntwk, U)
				# backward adjoints
				Λ = Network.adjoint(ntwk, X, label, loss)
				# store epoch F and G
				F += loss.evaluate(label, X, ntwk.results) / batch_size
				G += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type) / batch_size

				# append batch gradient
				method == "adjoint" ?
					append!(batch_grad, [paramGrad(loss, ntwk, U, label, syncFlag, partition)]) : # adjoint update
					append!(batch_grad, [paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type)]) # coadjoint update
			end # batch end

			# compute batch gradient
			update_grad = Helpers.batchGrad(batch_grad)

			# update ntwk params
			update_time = @elapsed begin
				adam_update!(ntwk, update_grad, η, syncFlag, partition, m, v, i)
			end # time end

			# stop training if F and/or G are numerically unstable (inf or NaN produced)
			Helpers.verifyNumericalStability(F,G) == true ? continue : return [epoch_info]
		end # epoch training samples end

		# classification error over test set
		error, test_F = RNN.add_test_error(ntwk, loss)
		# store epoch information
		epoch_info["F"][i] = F / n_train
		epoch_info["G"][i] = G / n_train
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = update_time
		λ_ind = RNN.rnnXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,10)
		epoch_info["Λ_init"][i] = norm(Λ[λ_ind[1:hid_dim]] ./ n_train)
		epoch_info["Λ_out"][i] = norm(Λ[λ_ind[end-hid_dim+1:end]] ./ n_train)
		#epoch_info["Error"][i] = error

		# WRITE EPOCH INFO TO FILE
		#epoch_file = "training_info/rnn_$(method)/sig_$(seed).dat"
	    #out_file = open(epoch_file, "a")
			printstyled("\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
			printstyled("\tError: $(error)\n", color=:light_magenta)
			printstyled("\tTest F Error: $(test_F)\n", color=:light_magenta)
			printstyled("\tLearning Rate: $(η)\n", color=:light_magenta)
			printstyled("\tObjective (F+G): $((F+G)/n_train)\n", color=:light_magenta)
			printstyled("\tF: $(F/n_train)\n", color=:light_magenta)
			printstyled("\tG: $(G/n_train)\n", color=:light_magenta)
			printstyled("\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
			printstyled("\t||λΤ||: $(epoch_info["Λ_out"][i])\n", color=:light_magenta)
			printstyled("\tTime: $(update_time)\n\n", color=:light_magenta)
		#close(f)

		#file_name = string("toy_exp/rnn_epochs/ntwks/rnn_$(method)_$(info_layer)_$(seed)_$(i).jld2")
		#save(file_name, "ntwk", ntwk)

	end # epoch end

	return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
# network structure
ntwk_type = "rnn"
seed = 2:3
experiment = "toy"
method = "adjoint"
hid_dim = 15
output_dim = 11
fc_dim = 0
inp_dim = 10 + hid_dim # corresponds to ntwk.features (input dim + inital hidden state dim)
seq_length = 10 # number of time points

# TRAINING HYPER PARAMS
η = 0.01 # learning rate
batch_size = 128
num_epochs = 50
lsmod = loss.crossEntropy
pnmod = penalty.var_phi

printstyled("Training Initialized: $(length(seed)) Ntwks Trained Sequentially.\n\n", color = :cyan)

for i=1:length(seed)
	ntwk, partition, syncFlag = RNN.specifyRNN(inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed[i])

	printstyled("**************\tTraining Method: RNN $(method)\t**************\n\n", color = :yellow, bold = :true)
	#printstyled("\tInformation Layer: $(info_layer)\n", color = :light_cyan)
	printstyled("\tSeed: $(seed[i])\n", color = :light_cyan)
	printstyled("\tBatch Size: $(batch_size)\n", color = :light_cyan)
    printstyled("\tLearning Rate: $(η)	\n", color = :light_cyan)
    printstyled("\tLoss (F): Cross Entropy (Softmax)\n", color = :light_cyan)
	printstyled("\tPenalty (G): Var Phi\n", color = :light_cyan)
	printstyled("\tNetwork Architecture:\n", color=:light_cyan)
	printstyled("\t\tInput Dim: $(inp_dim) (10 + $(hid_dim) init states)\n", color = :light_cyan)
	printstyled("\t\tHidden Dim: $(hid_dim)\n", color = :light_cyan)
	printstyled("\t\tFully Connect Dim: $(fc_dim)\n", color = :light_cyan)
	printstyled("\t\tSoftmax Dim: $(output_dim)\n\n", color = :light_cyan)
	printstyled("**************************************************************\n\n", color = :yellow, bold = :true)
	# train network
	training_output = train(ntwk, lsmod, pnmod, η, num_epochs, method, syncFlag, partition, seed[i], batch_size, ntwk_type)

	# save trained network
	#file_name = string("toy_exp/rnn_$(method)_$(info_layer)_$(seed[i]).jld")
	#save(file_name, "ntwk", ntwk)
	#ntwk_test = load("ntwk_test.jld", "ntwk")
	printstyled("Ntwks Trained: $(i/length(seed)*100)% ...\n\n", color=:cyan)
end
