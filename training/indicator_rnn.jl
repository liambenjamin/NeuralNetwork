include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/lstm.jl")
include("../src/gru.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")


using FileIO, NPZ, LinearAlgebra, CSV, JLD2, SparseArrays, Random, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU


"""
Implements RNN Training on Toy Experiment
"""
function train(ntwk, loss :: Module, penalty :: Module, η :: Float64, num_epochs :: Int64, method :: String, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, batch_size :: Int64, ntwk_type :: String, optimizer :: String)

	epoch_info = Dict("F" => zeros(num_epochs), "G" => zeros(num_epochs), "Epoch" => zeros(num_epochs),
                        "Time" => zeros(num_epochs), "Error" => zeros(num_epochs), "Λ_init" => zeros(num_epochs),
                        "Λ_out" => zeros(num_epochs), "Test_F" => zeros(num_epochs), "Test_G" => zeros(num_epochs),
						"Test_Λ_init" => zeros(num_epochs), "Test_Λ_out" => zeros(num_epochs)
						)

	# initialize adam running exponential average vars
	m, v = Helpers.adam_init(ntwk, syncFlag, partition)

	test_x, test_y = Helpers.generate_dataset(2000, ntwk.seq_length, 2)

	for i=1:num_epochs

		# epoch totals
		F = 0.0
		G = 0.0
		Λ = zeros(size(ntwk.graph,1))
		t = 0.0
		n_batches = 150
		for j=1:n_batches #iterate through batches
			t += @elapsed begin
				batch_x, batch_y = Helpers.generate_batch(batch_size, ntwk.seq_length, 2)
				# updates network and returns batch summary (avg)

				b_F, b_G, b_Λ = Helpers.evaluate_batch(ntwk, batch_x, batch_y, loss, penalty, η, syncFlag, partition, optimizer, method, ntwk_type, m, v, i, batch_size)
				F += b_F
				G += b_G
				Λ += b_Λ
			end # end elapsed

			# print update every 50 training batches
			if j == 1 || j % 50 == 0 || j == n_batches
				printstyled("\r\tEpoch $(i) ~ Batch [$(j)/$(n_batches)] ~ Epoch Duration:  $(t)", color=:light_cyan, bold=:true)
			end
		end # end training batches

		λ_ind = Helpers.ntwkX_ind(ntwk, ntwk_type)

		# returns average values
		test_F, test_G, test_Λ, test_error = Helpers.eval_binary_loss(ntwk, test_x, test_y, loss, penalty, ntwk_type)
		# classification error over test set
		#test_F, test_error = Helpers.mnist_test_error(ntwk, test_x, test_y, loss, ntwk_type)
		epoch_info["Error"][i] = test_error
		epoch_info["F"][i] = F / n_batches # batch avg
		epoch_info["G"][i] = G / n_batches # batch avg
		epoch_info["Test_F"][i] = test_F
		epoch_info["Test_G"][i] = test_G
		epoch_info["Epoch"][i] = i
		epoch_info["Time"][i] = t / n_batches # batch avg
		epoch_info["Λ_init"][i] = norm(Λ[λ_ind[1:ntwk.hid_dim]]) / n_batches
		epoch_info["Λ_out"][i] = norm(Λ[λ_ind[end-ntwk.hid_dim+1:end]]) / n_batches
		epoch_info["Test_Λ_init"][i] = norm(test_Λ[λ_ind[1:ntwk.hid_dim]])
		epoch_info["Test_Λ_out"][i] = norm(test_Λ[λ_ind[end-ntwk.hid_dim+1:end]])

		printstyled("\n\tError: $(test_error)\n", color=:light_magenta)
		printstyled("\tTrain F: $(F/n_batches)\n", color=:light_magenta)
		printstyled("\tTrain G: $(G/n_batches)\n", color=:light_magenta)
		printstyled("\tTest F: $(test_F)\n", color=:light_magenta)
		printstyled("\tTest G: $(test_G)\n", color=:light_magenta)
		printstyled("\tTrain ||λ₀||: $(epoch_info["Λ_init"][i])\n", color=:light_magenta)
		printstyled("\tTrain ||λT||: $(epoch_info["Λ_out"][i])\n\n", color=:light_magenta)

		# save intermediate ntwk after epoch i
		file_name = string("../analysis/toy/$(ntwk_type)_$(method)_1e2_$(i).jld2")
		save(file_name, "ntwk", ntwk)
	end # end epoch

	return [epoch_info]
end

# TRAINING PARAMETERS/CONSTANTS
ntwk_type = "rnn"
method = "coadjoint"
seed = 1 #parse(Int64, ENV["seed"])
hid_dim = 10
output_dim = 1
inp_dim = 20 # flattened feature image (28 × 28)
fc_dim = 0 # dimension of fully connect layer (optional)
seq_length = 20 # number of time steps in sequence
η = 0.001 # learning rate default for adam (sgd = 0.01)
batch_size = 32
num_epochs = 20
lsmod = loss.binary_crossEntropy # loss
pnmod = penalty.var_phi # adjoint control function
opt = "adam"

# intiailize network, partition and verify
ntwk, partition, syncFlag = Helpers.specify_ntwk(ntwk_type, inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)

d = string("../analysis/toy/$(ntwk_type)_randomInit2.jld2")
save(d, "ntwk", ntwk)

# load ntwk weights
KERAS_WEIGHTS = npzread("toy-params/keras-rnn-toy-adam.npz")
W_REC_K = vcat(KERAS_WEIGHTS["w_inp"],KERAS_WEIGHTS["w_rec"])
W_REC_BIAS_K = KERAS_WEIGHTS["w_rec_bias"]
W_OUT_K = KERAS_WEIGHTS["w_out"]
W_OUT_BIAS_K = KERAS_WEIGHTS["w_out_bias"]

# load ntwk gradients
KERAS_GRADS = npzread("toy-params/keras-rnn-toy-adam-grad.npz")
dW_REC_K = vcat(KERAS_GRADS["dw_inp"],KERAS_GRADS["dw_rec"])
dW_REC_BIAS_K = KERAS_GRADS["dw_rec_bias"]
dW_OUT_K = KERAS_GRADS["dw_out"]
dW_OUT_BIAS_K = KERAS_GRADS["dw_out_bias"]

# seed network with keras pretrained parameters
ntwk = Helpers.set_rnn_params(ntwk, partition, W_REC_K, W_REC_BIAS_K, W_OUT_K, W_OUT_BIAS_K)
# load and transform input-output pair
VERIFY_PAIR = npzread("toy-params/keras-rnn-toy-test-pair.npz")
label = VERIFY_PAIR["label"]
U = collect(Iterators.flatten(vcat(VERIFY_PAIR["U"], zeros(ntwk.hid_dim))))
#label = convert.(Float64, map(i -> i==5 ? 1 : 0, 0:9))
# evaluate gradient
dP = paramGrad(lsmod, ntwk, U, label, syncFlag, partition)

# verify gradients for first neuron
julia_grad_rec = dP[partition[1][end]] # note: 541 is max index in partition containing neuron 1
keras_grad_rec = vcat(dW_REC_BIAS_K[1], dW_REC_K[:,1])
# verify gradients for output neurons
julia_grad_out = dP[partition[end][1]] # note: 561 is first index in partition containing output (sigmoid neuron)
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
printstyled("Starting Training on Batched Indicator Task:\n\tTraining Method:\t$(method)\n\tOptimizer:\t\t$(opt)\n\n", color=:yellow)
#training_output = train(ntwk, lsmod, pnmod, η, num_epochs, method, syncFlag, partition, batch_size, ntwk_type, opt)
#f = string("../analysis/toy/$(ntwk_type)_$(method)_1e2_output.jld2")
#save(f, "output", training_output)
