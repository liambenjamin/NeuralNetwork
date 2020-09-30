# training script for toy experiment
η = 0.01
seed = parse(Int64,ENV["seed"])

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/penalties.jl")
include("../src/helpers.jl")
include("../src/gru.jl")


using FileIO, LinearAlgebra, HDF5, NPZ, JLD2, SparseArrays, Random, Colors, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.GRU

function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag, partition, ntwk_type :: String)

    # store adjoint and coadjoint training epoch summary
    epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
						"Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
						"Λ_out" => zeros(num_epochs))

    test_x = npzread("data/test_imdb_dat_1.npz")
    test_y = npzread("data/test_imdb_label1.npz")

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)
	batch_size = 20
	N_samp = 25000

    for i=1:num_epochs
		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))
		# time sgd update
		#update_time = 0.0

            for j=1:5 # number of splits in training set
                prefix = "data/"
                name = string("train_imdb_dat_", j, ".npz")
                label_name = string("train_imdb_label", j, ".npz")
                feats = npzread(prefix * name)
                labels = npzread(prefix * label_name)
				# count batch size
				ct = 0
                for l=1:size(feats,1)

					batch_grad = []

					U, label = collect(Iterators.flatten(feats[l,:,:])), zeros(ntwk.results)
					labels[l] > 0 ? label[1] = 1.0 : label[2] = 1.0

					U = vcat(U, zeros(ntwk.hid_dim))
					X = Network.evaluate(ntwk, U)
					Λ = Network.adjoint(ntwk, X, label, loss)
					F += loss.evaluate(label, X, ntwk.results) / batch_size
					G += penalty.evaluate(Λ, ntwk.features, ntwk.hid_dim, ntwk.seq_length, ntwk_type) / batch_size

					# append batch gradient
					method == "adjoint" ?
						append!(batch_grad, [paramGrad(loss, ntwk, U, label, syncFlag, partition)]) : # adjoint update
						append!(batch_grad, [paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type)]) # coadjoint update
					ct += 1

					if ct == batch_size

						# compute batch gradient
						update_grad = Helpers.batchGrad(batch_grad)

						# update ntwk params
						adam_update!(ntwk, update_grad, η, syncFlag, partition, m, v, i)

						# stop training if F and/or G are numerically unstable (inf or NaN produced)
						Helpers.verifyNumericalStability(F,G) == true ? continue : return [epoch_info]
						# reset batch_grad and ct
						batch_grad = []
						ct = 0
					end # end batch update
				end # end 1/5 training data
            end # end epoch training

			# classification error over test set
			error = GRU.classification_error(ntwk, test_x, test_y, true)

			epoch_info["F"][i] = F / N_samp
			epoch_info["G"][i] = G / N_samp
			epoch_info["Epoch"][i] = i
			#epoch_info["Time"][i] = update_time
			λ_ind = GRU.gruXind(ntwk.features-ntwk.hid_dim,ntwk.hid_dim,ntwk.seq_length)
			epoch_info["Λ_init"][i] = norm(Λ[λ_ind[1:hid_dim]] ./ N_samp)
			epoch_info["Λ_out"][i] = norm(Λ[λ_ind[end-hid_dim+1:end]] ./ N_samp)
			epoch_info["Error"][i] = error

			# WRITE EPOCH INFO TO FILE
			epoch_file = "training-info/gru/gru_$(method)_$(seed).dat"
		    out_file = open(epoch_file, "a")
				printstyled(out_file, "\tEpoch: $(i)\n", color=:light_cyan, bold=:true)
				printstyled(out_file, "\tError: $(error)\n", color=:light_magenta)
				printstyled(out_file, "\tObjective (F+G): $(F+G)\n", color=:light_magenta)
				printstyled(out_file, "\tF: $(F/N_samp)\n", color=:light_magenta)
				printstyled(out_file, "\tG: $(G/N_samp)\n", color=:light_magenta)
				printstyled(out_file, "\t||λ0||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
				printstyled(out_file, "\t||λΤ||: $(epoch_info["Λ_out"][i])\n", color=:light_magenta)
				#printstyled(out_file, "\tTime: $(update_time)\n\n", color=:light_magenta)
			close(out_file)
			# save intermediate ntwk after epoch i
			file_name = string("ntwks/gru/gru_$(method)_$(seed)_$(i).jld2")
			save(file_name, "ntwk", ntwk)

		end # end epoch

    return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
# network structure
ntwk_type = "gru"
method = "adjoint"
hid_dim = 10
output_dim = 2
inp_dim = 50*100 + hid_dim # corresponds to ntwk.features (input dim + 2*inital hidden state dim)
fc_dim = 0 # dimension of fully connect layer post recursive layers
seq_length = 50 # number of time points (dimension of U)
ntwk, partition, syncFlag = GRU.specifyGRU(inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)
η = 0.01 # learning rate
num_epochs = 100
lsmod = loss.crossEntropy
pnmod = penalty.var_phi

# write network training parameters to file
epoch_file = "training-info/gru/gru_$(method)_$(seed).dat"


output_file = open(epoch_file, "a")
	printstyled(output_file, "**************\tTraining Method: GRU $(method)\t**************\n\n", color = :yellow, bold = :true)
    printstyled(output_file, "\tSeed: $(seed)\n", color = :light_cyan)
    printstyled(output_file, "\tLearning Rate: $(η)\n", color = :light_cyan)
    printstyled(output_file, "\tLoss (F): Cross Entropy (Softmax)\n", color = :light_cyan)
	printstyled(output_file, "\tPenalty (G): Var + ϕ\n", color = :light_cyan)
	printstyled(output_file, "\tNetwork Architecture:\n", color=:light_cyan)
	printstyled(output_file, "\t\tInput Dim: $(inp_dim)\n", color = :light_cyan)
	printstyled(output_file, "\t\tHidden Dim: $(hid_dim)\n", color = :light_cyan)
	printstyled(output_file, "\t\tFully Connect Dim: $(fc_dim)\n", color = :light_cyan)
	printstyled(output_file, "\t\tSoftmax Dim: $(output_dim)\n\n", color = :light_cyan)
	printstyled(output_file, "***************************************************************\n\n", color = :yellow, bold = :true)
close(output_file)

# train ntwk
training_output = train(ntwk, lsmod, pnmod, η, num_epochs, method, syncFlag, partition, ntwk_type)

# save trained network
file_name = string("ntwks/gru/gru_$(method)_trained.jld2")
save(file_name, "ntwk", ntwk)
