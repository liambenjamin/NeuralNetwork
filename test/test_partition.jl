include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")
include("../src/lstm.jl")

using LinearAlgebra, SparseArrays, Main.Neurons, Main.Network, Main.Partition, Main.Helpers, Main.loss, Main.penalty, Main.LSTM

#Experimental architectures and partitions
testNets = Function[
#=function makeMNISTSoftmax()

    hid_dim = 10
    output_dim = 10
    input_dim = 56 + hid_dim
    seq_length = 2
    fc_dim = 10
    seed = 1

    neurons, rowInd, colInd = Helpers.specifyGraph(input_dim, hid_dim, output_dim, fc_dim, seq_length, seed)
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd,colInd,vals)
    ntwk = Network.network(neurons, input_dim, output_dim, hid_dim, fc_dim, seq_length) # (neuron list, input dim, output dim)
    Network.graph!(ntwk,graph)

    #neurons, rowInd, colInd = Helpers.FFN(input_dim, hid_dim, output_dim, fc_dim, seq_length, 1)
    #ntwk = specifyFFN(input_dim, output_dim, hid_dim, fc_dim, seq_length, seed)

    # partition neurons to share weights
    partition = Helpers.getPartition(length(ntwk.neurons), hid_dim, output_dim, fc_dim, seq_length)
    Partition.synchronizeParameters!(ntwk,partition)
    # verify partition
    syncFlag = Partition.verifyPartition(ntwk,partition)

    #Name
    name = "MNIST RNN (28×2 input) w/ Softmax Output (10 classes) Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t $(input_dim)-Dimen ($(input_dim-hid_dim) for input and $(hid_dim) for hidden state)
    \t|||||\t |||||
    \tFull Co.\t |||||
    \tNeuron  \t Sigmoid
    \t|||||\t |||||
    \tSoftmax Full Co.\t |||||
    \t|||||\t |||||
    \tOutput  \t $(output_dim)-Dimen \n\n
    """
    label = zeros(10) * 0.1
    label[5] = 1.0
    return name, desc, ntwk, partition, 0.01*randn(input_dim), label
end,
=#
function makeTestAutoLSTM()

    #Implements specified LSTM architecture using `lstm.jl`

    feat_dim = 6
    hid_dim = 3
    inp_dim = feat_dim + 2*hid_dim
    fc_dim = 0
    seq_length = 2
    output_dim = 3
    seed = 2
    label = [1.0, 0.0, 0.0]
    ntwk, partition, syncFlag = LSTM.specifyLSTM(inp_dim, output_dim, hid_dim, fc_dim, seq_length, seed)



    #Name
    name = "Test Auto Generate LSTM Network"

    #Description
    desc = """
    \tNetwork Description
    \t|||||\t |||||
    \tFeature \t $(feat_dim)-Dimen: $(feat_dim-2*hid_dim) (feat) + $(2*hid_dim) (x₀ and c₀)
    \t|||||\t |||||
    \tHidden Dim \t $(hid_dim)
    \t|||||\t |||||
    \tSequence Length\t $(seq_length)
    \t|||||\t |||||
    \tNeuron  \t Sigmoid or tanH (Dependent on Gating Unit)
    \t|||||\t |||||
    \tFull Co.\t Sigmoid (# of neurons correspond to # of output categories)
    \t|||||\t |||||
    \tOutput  \t $(output_dim)-Dimen (Softmax) \n\n
    """

    return name, desc, ntwk, partition, 0.01*randn(inp_dim), label
end

]
#=
function makeTwoLayerLogisticLogistic()

    #Assumes 2 dimensional input
    #Both inputs feed into 2 logistic neurons with identical parameters
    #Outputs from both logistic neurons feed into 2 more logistic neurons with identical parameters
    #Outputs 2 values, one from each logistic neuron

    neurons = Vector{neuron}(undef,4)
    neurons[1] = logistic.init(1,2,4)
    neurons[2] = logistic.init(2,2,4)
    neurons[3] = logistic.init(3,2,4)
    neurons[4] = logistic.init(4,2,4)

    rowInd = [1,2,1,2,3,4,3,4,5,6]
    colInd = [1,1,2,2,3,3,4,4,5,5]
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd,colInd,vals)

    ntwk = Network.network(neurons,2,2)
    Network.graph!(ntwk,graph)

    partition = [[1,2], [3,4]]
    Partition.synchronizeParameters!(ntwk,partition)

    #Name
    name = "Two-layer Logistic-Logistic Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 2-Dimen
    \tFull Co \t |||||||
    \tLayer   \t 2 Logistic \t == Parameters
    \tFull Co \t |||||||
    \tLayer   \t 2 Logistic \t == Parameters
    \tPart Co \t |  |  |
    \tOutput  \t 2-Dimen\n\n
    """

    return name, desc, ntwk, partition, 0.01*randn(2)
end,
function makeConvolutionSmoothMax()

    #Assumes 10 dimensional input
    #Inputs feed into a layer of 2 convolution neurons with identical parameters
    #Layer outputs feed into a single smoothmax function
    #Outputs 1 dimension value from smoothmax function

    neurons = Vector{neuron}(undef,3)
    neurons[1] = convolution.init(1,10,3,met=[3,0,3])
    neurons[2] = convolution.init(2,10,3,met=[3,0,3])
    neurons[3] = smoothmax.init(3,4,1)

    rowInd = vcat(1:10, 1:10, 11:14,15)
    colInd = vcat(1*ones(Int64,10), 2*ones(Int64,10), 3*ones(Int64,4), 4)
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd, colInd, vals)

    ntwk = Network.network(neurons,10,1) #Neurons, Input dimenion, Output dimension
    Network.graph!(ntwk,graph)

    partition = [[1,2],[3]]
    Partition.synchronizeParameters!(ntwk,partition)
    #Name
    name = "Two-layer Convolution-SmoothMax Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 10-Dimen
    \tFull Co \t |||||||
    \tLayer   \t 2 1-d-Conv \t == Parameters
    \tFull Co \t |||||||
    \tNeuron  \t Softmax
    \tFull Co \t |||||||
    \tOutput  \t 1-Dimen\n\n
    """

    return name, desc, ntwk, partition, 0.01*randn(10)
end
]
=#

#Numerical Derivatives
function numerical_parameter_adjoint(loss :: Module, ntwk, label, partition, feat; ϵ= 1e-8)
    numDer = Dict{Int64,Vector{Float64}}(maximum(class) => zeros(ntwk.neurons[class[1]].par) for class in partition)
    Partition.synchronizeParameters!(ntwk,partition)
    X = Network.evaluate(ntwk,feat)
    f_bse = loss.evaluate(label, X, ntwk.results)
    for class in partition
        ntwk_copy = deepcopy(ntwk)
        max_j = maximum(class)
        for j = 1:ntwk.neurons[max_j].par
            ntwk_copy.neurons[max_j].β[j] += ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
            X_copy = Network.evaluate(ntwk_copy,feat)
            numDer[max_j][j] = (loss.evaluate(label, X_copy, ntwk.results) - f_bse)/ϵ
            ntwk_copy.neurons[max_j].β[j] -= ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
        end
    end
    return numDer
end

function numerical_parameter_adjoint(loss :: Module, ntwk, partition, feat, label; ϵ= 1e-8)
    numDer = Dict{Int64,Vector{Float64}}(maximum(class) => zeros(ntwk.neurons[class[1]].par) for class in partition)
    Partition.synchronizeParameters!(ntwk,partition)
    X = Network.evaluate(ntwk,feat)
    f_bse = loss.evaluate(label,X, ntwk.results)
    for class in partition
        ntwk_copy = deepcopy(ntwk)
        max_j = maximum(class)
        for j = 1:ntwk.neurons[max_j].par
            ntwk_copy.neurons[max_j].β[j] += ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
            X_copy = Network.evaluate(ntwk_copy,feat)
            numDer[max_j][j] = (loss.evaluate(label,X_copy,ntwk.results) - f_bse)/ϵ
            ntwk_copy.neurons[max_j].β[j] -= ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
        end
    end
    return numDer
end

function numerical_parameter_coadjoint(loss :: Module, penalty :: Module, ntwk, partition, feat, label; ϵ = 1e-8)
    numDer = Dict{Int64,Vector{Float64}}(maximum(class) => zeros(ntwk.neurons[class[1]].par) for class in partition)
    Partition.synchronizeParameters!(ntwk,partition)
    X = Network.evaluate(ntwk,feat)
    λ = Network.adjoint(ntwk, X, label, loss)
    bse = loss.evalutate(label,X,ntwk.results) + penalty.evaluate(λ,ntwk.features,ntwk.hid_dim,ntwk.seq_length)
    for class in partition
        ntwk_copy = deepcopy(ntwk)
        max_j = maximum(class)
        for j = 1:ntwk.neurons[max_j].par
            ntwk_copy.neurons[max_j].β[j] += ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
            X_copy = Network.evaluate(ntwk_copy,feat)
            λ_copy = Network.adjoint(ntwk_copy, X_copy, label, loss)
            numDer[max_j][j] = (loss.evaluate(label,X_copy,ntwk.results) + penalty.evaluate(λ_copy, ntwk.features,ntwk.hid_dim,ntwk.seq_length) - bse)/ϵ
            ntwk_copy.neurons[max_j].β[j] -= ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
        end
    end
    return numDer
end

function numerical_parameter_coadjoint(loss :: Module, penalty :: Module, ntwk, partition, feat, label; ϵ = 1e-8)
    numDer = Dict{Int64,Vector{Float64}}(maximum(class) => zeros(ntwk.neurons[class[1]].par) for class in partition)
    Partition.synchronizeParameters!(ntwk,partition)
    X = Network.evaluate(ntwk,feat)
    λ = Network.adjoint(ntwk, X, label, loss)
    bse = loss.evaluate(label,X,ntwk.results) + penalty.evaluate(λ,ntwk.features,ntwk.hid_dim,ntwk.seq_length)
    for class in partition
        ntwk_copy = deepcopy(ntwk)
        max_j = maximum(class)
        for j = 1:ntwk.neurons[max_j].par
            ntwk_copy.neurons[max_j].β[j] += ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
            X_copy = Network.evaluate(ntwk_copy,feat)
            λ_copy = Network.adjoint(ntwk_copy, X_copy, label, loss)
            numDer[max_j][j] = (loss.evaluate(label,X_copy,ntwk.results) + penalty.evaluate(λ_copy,ntwk.features,ntwk.hid_dim,ntwk.seq_length) - bse)/ϵ
            ntwk_copy.neurons[max_j].β[j] -= ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
        end
    end
    return numDer
end

#Partition parsing helper
function getPartitionMax(i, partition)
    for class in partition
        (i in class) && return maximum(class)
    end
end

printstyled("****************************************\nBeginning unit tests for partitions\n****************************************\n\n"; color=:light_magenta)
lsmod = loss.crossEntropy
pnmod = penalty.log_L2
for networkEncap in testNets
    nme, desc, ntwk, partition, feat, label = networkEncap()

    printstyled("Testing: ",color=:light_magenta);
    printstyled("$nme\n\n"; bold=true, color=:blue)
    printstyled(desc, color=:blue)

    #Verify Partition
    printstyled("Verifying partition\t\t\t\t\t")
    syncFlag = Partition.verifyPartition(ntwk,partition)
    syncFlag ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Test Adjoint Parameter Derivative
    printstyled("Computing (adjoint-sync) parameter derivative\t\t")
    grads = paramGrad(lsmod, ntwk, feat, label, syncFlag, partition)
    printstyled("COMPLETED\n",color=:light_yellow)

    printstyled("Computing (adjoint) numerical parameter derivative\t")
    gradNum = numerical_parameter_adjoint(lsmod, ntwk, partition, feat, label)
    printstyled("COMPLETED\n",color=:light_yellow)

    printstyled("Comparing (adjoint) parameter derivatives\t\t")
    maximum([norm( grads[k] - gradNum[k]) for k in keys(grads)]) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);
    printstyled("$(maximum([norm( grads[k] - gradNum[k]) for k in keys(grads)]))\n", color=:blue)

    #Verify update!
    printstyled("Testing (adjoint) update function\t\t\t")
    L = length(ntwk.neurons)
    step = 0.1
    copy_ntwk = deepcopy(ntwk)
    old_param = Dict(i => copy_ntwk.neurons[i].β for i =1:L)
    dB = Dict(i =>  grads[getPartitionMax(i,partition)] for i = 1:L)
    upd_param = Dict(i => old_param[i]-step*dB[i] for i = 1:L)
    update!(lsmod, copy_ntwk, feat, label, step, syncFlag, partition)
    maximum([norm(copy_ntwk.neurons[k].β - upd_param[k]) for k in 1:L] ) < 1e-12 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Test Coadjoint Parameter Derivative
    printstyled("Computing (coadjoint-sync) parameter derivative\t\t")
    grads = paramGrad(lsmod, pnmod, ntwk, feat, label, syncFlag, partition)
    #paramGrad(df :: Function,d2f :: Function, dg :: Function, ntwk :: Network.network, feat :: Vector{Float64}, syncFlag :: Bool, partition = Vector{Vector{Int64}})
    #partition = Vector{Vector{Int64}}

    printstyled("COMPLETED\n",color=:light_yellow)

    printstyled("Computing (coadjoint) numerical parameter derivative\t")
    gradNum = numerical_parameter_coadjoint(lsmod, pnmod, ntwk, partition, feat, label)
    printstyled("COMPLETED\n",color=:light_yellow)

    printstyled("Comparing (coadjoint) parameter derivatives\t\t")
    maximum([norm( grads[k] - gradNum[k]) for k in keys(grads)]) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);
    printstyled("$(maximum([norm( grads[k] - gradNum[k]) for k in keys(grads)]))\n", color=:blue)
    #Verify update!
    printstyled("Testing (coadjoint) update function\t\t\t")
    L = length(ntwk.neurons)
    step = 0.1
    copy_ntwk = deepcopy(ntwk)
    old_param = Dict(i => copy_ntwk.neurons[i].β for i =1:L)
    dB = Dict(i =>  grads[getPartitionMax(i,partition)] for i = 1:L)
    upd_param = Dict(i => old_param[i]-step*dB[i] for i = 1:L)
    update!(lsmod, pnmod, copy_ntwk, feat, label, step, syncFlag, partition)
    maximum([norm(copy_ntwk.neurons[k].β - upd_param[k]) for k in 1:L] ) < 1e-12 ?
        printstyled("PASSED\n\n", color=:green) :
        printstyled("FAILED\n\n", color=:red);
end
