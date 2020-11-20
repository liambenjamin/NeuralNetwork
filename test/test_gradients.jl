#Author: Liam Johnston
#Date: 11-19-2020

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/lstm.jl")
include("../src/gru.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")

using FileIO, NPZ, LinearAlgebra, SparseArrays, MLDatasets, Random, Main.Neurons, Main.Network, Main.loss, Main.penalty, Main.Helpers, Main.Partition, Main.RNN, Main.LSTM, Main.GRU

"""
Load Trained MNIST network
"""

printstyled("\tLoading trained network...", color=:yellow)
# TRAINING PARAMETERS/CONSTANTS
ntwk_type = "rnn"
method = "adjoint"
seed = 1 #parse(Int64, ENV["seed"])
hid_dim = 20
output_dim = 10
inp_dim = 784 # flattened feature image (28 × 28)
fc_dim = 0 # dimension of fully connect layer (optional)
seq_length = 28 # number of time steps in sequence
η = 0.001 # learning rate default for adam (sgd = 0.01)
batch_size = 32
num_epochs = 5
lsmod = loss.softmaxCrossEntropy # loss
pnmod = penalty.var_phi # adjoint control function
opt = "adam"

# intiailize network, partition and verify
ntwk, partition, syncFlag = Helpers.specify_ntwk(ntwk_type, inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)
# load ntwk weights
KERAS_WEIGHTS = npzread("../training/mnist-params/keras-rnn-mnist-adam-four.npz")
W_REC_K = vcat(KERAS_WEIGHTS["w_inp"],KERAS_WEIGHTS["w_rec"])
W_REC_BIAS_K = KERAS_WEIGHTS["w_rec_bias"]
W_OUT_K = KERAS_WEIGHTS["w_out"]
W_OUT_BIAS_K = KERAS_WEIGHTS["w_out_bias"]
# load ntwk gradients
KERAS_GRADS = npzread("../training/mnist-params/keras-rnn-mnist-adam-grad-four.npz")
dW_REC_K = vcat(KERAS_GRADS["dw_inp"],KERAS_GRADS["dw_rec"])
dW_REC_BIAS_K = KERAS_GRADS["dw_rec_bias"]
dW_OUT_K = KERAS_GRADS["dw_out"]
dW_OUT_BIAS_K = KERAS_GRADS["dw_out_bias"]
# seed network with keras pretrained parameters
ntwk = Helpers.set_rnn_params(ntwk, partition, W_REC_K, W_REC_BIAS_K, W_OUT_K, W_OUT_BIAS_K)
printstyled("completed.\n", color=:green)
# Load testing set
printstyled("\tLoading data...", color=:yellow)
test_x, test_y = Helpers.load_mnist_test()
test_x, test_y = test_x[1:5], test_y[1:5,:]
printstyled("completed.\n", color=:green)

printstyled("\tComputing gradients of $(length(test_x)) samples...", color=:yellow)
F, G, Λ, grads = Helpers.inputGradients(ntwk, lsmod, pnmod, test_x, test_y, "adjoint", "rnn")
printstyled("completed.\n\n", color=:green)


"""
Numerical gradient wrt input
"""
function numerical_input_gradient(ntwk :: Network.network, test_x, test_y; ϵ=1e-8)
    grad_mat = zeros(size(test_x[1],1), size(test_x,1))
    for i=1:length(test_x) #5
        U, label = test_x[i], test_y[i,:]
        N = length(U)
        U = vcat(U, zeros(ntwk.hid_dim))
        X = Network.evaluate(ntwk,U)
        f_bse = lsmod.evaluate(label, X, ntwk.results)

        for j=1:N # length of feature vector
            pert = zeros(length(U))
            pert[j] = ϵ
            U_pert = U+pert
            X_pert = Network.evaluate(ntwk,U_pert)
            f_pert = lsmod.evaluate(label, X_pert, ntwk.results)
            grad_mat[j,i] = (f_pert - f_bse) / ϵ
        end
    end
    return grad_mat
end

printstyled("\tTesting Numerical Gradient with respect to the input...\n", color=:cyan)

numerical_grads = numerical_input_gradient(ntwk, test_x, test_y)
dU = Helpers.all_input_grads(ntwk, grads, partition)

for i=1:size(numerical_grads,2)
    printstyled("\t\tTesting Sample $(i):", color=:cyan)
    norm(dU[:,i]-numerical_grads[:,i]) < 1e-5 ?
        printstyled("\tPASS\n", color=:green) :
        printstyled("\tFAIL\n", color=:red)
end
