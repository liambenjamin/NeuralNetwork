#Various tests for verification of LSTM functions
#Author: Liam Johnston
#Date: August 25, 2020

include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/lstm.jl")

using Colors, Random, LinearAlgebra, SparseArrays, Main.Neurons, Main.Network, Main.Partition, Main.LSTM
printstyled("\n***************************************\n",color=:yellow)
printstyled("\tTesting LSTM Functions\n", color=:yellow)
printstyled("***************************************\n\n",color=:yellow)
################################################################################

printstyled("getFeatureIndex():\n", color=:magenta)


result1 = [[1 4], [5 8]]
func_result1 = LSTM.getFeatureIndex(8, 2)

result2 = [[1 4], [5 8], [9 12]]
func_result2 = LSTM.getFeatureIndex(12,3)

result3 = [[1 3], [4 6], [7 9], [10 12]]
func_result3 = LSTM.getFeatureIndex(12,4)

result1 == func_result1 ?
    printstyled("\t\t\t\tPASS\n", color=:green) :
    printstyled("\t\t\t\tFAIL\n", color=:red)


result2 == func_result2 ?
    printstyled("\t\t\t\tPASS\n", color=:green) :
    printstyled("\t\t\t\tFAIL\n", color=:red)


result3 == func_result3 ?
    printstyled("\t\t\t\tPASS\n", color=:green) :
    printstyled("\t\t\t\tFAIL\n", color=:red)


################################################################################

printstyled("getStateIndex():\n", color=:magenta)

state_ind1, prev_x = LSTM.getStateIndex(4, 1, 2)

result1 = [[5], [5], [5], [5], [7, 6, 8, 10], [9, 11], [12], [12], [12], [12], [13, 11, 14, 16], [15, 17]]

state_ind2, prev_x2 = LSTM.getStateIndex(6, 2, 2)

result2 = [[7,8], [7,8], [7,8], [7,8], [7,8], [7,8], [7,8], [7,8], [11,12,9,10,13,14,17,18],
            [15,16,19,20], [21,22], [21,22], [21,22], [21,22], [21,22], [21,22], [21,22], [21,22],
            [23,24,19,20,25,26,29,30],[27,28,31,32]]


result1 == state_ind1 ?
    printstyled("\t\t\t\tPASS\n", color=:green) :
    printstyled("\t\t\t\tFAIL\n", color=:red)


result2 == state_ind2 ?
    printstyled("\t\t\t\tPASS\n", color=:green) :
    printstyled("\t\t\t\tFAIL\n", color=:red)

################################################################################

printstyled("lstmNeurons():\n", color=:magenta)

feat_dim = 4
hid_dim = 1
seq_length = 2
output_dim = 4
seed = 1

neurons = LSTM.lstmNeurons(4,1,2,4,1)

Random.seed!(1)
neuron_hc = [sigmoid.init(1,3,4), sigmoid.init(2,3,4), sigmoid.init(3,3,4),
                    sigmoid.init(4,3,4), lstmCellState.init(5,4,5), lstmHiddenState.init(6,2,3),
                    sigmoid.init(7,3,4), sigmoid.init(8,3,4), sigmoid.init(9,3,4),
                    sigmoid.init(10,3,4), lstmCellState.init(11,4,5), lstmHiddenState.init(12,2,3),
                    sigmoid.init(13,1,2), sigmoid.init(14,1,2), sigmoid.init(15,1,2),
                    sigmoid.init(16,1,2), softmax.init(17,4,1)
                    ]
maximum([norm(neurons[k].β - neuron_hc[k].β) for k in 1:length(neurons)]) < 1e-7 ?
    printstyled("\t\t\t\tPASS\n", color=:green) :
    printstyled("\t\t\t\tFAIL\n", color=:red)
