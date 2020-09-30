include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")
include("../src/rnn2.jl")

using LinearAlgebra, SparseArrays, Main.Neurons, Main.Network, Main.Partition, Main.Helpers, Main.loss, Main.penalty, Main.RNN

#Experimental architectures and partitions
testNets = Function[
function makeMNISTSoftmax()

    hid_dim = 5
    output_dim = 10
    input_dim = 784 + hid_dim
    seq_length = 28
    fc_dim = 0
    seed = 2

    ntwk, partition, syncFlag = RNN.specifyRNN(input_dim, output_dim, hid_dim, fc_dim, seq_length, seed)

    #Name
    name = "MNIST RNN (28×28 input) w/ Softmax Output (10 classes) Network"

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
    label = zeros(10)
    label[5] = 1.0
    ntwk_type = "rnn"
    return name, desc, ntwk, partition, randn(input_dim), label, ntwk_type
end
]


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

function numerical_parameter_coadjoint(loss :: Module, penalty :: Module, ntwk, partition, feat, label, ntwk_type; ϵ = 1e-8)
    numDer = Dict{Int64,Vector{Float64}}(maximum(class) => zeros(ntwk.neurons[class[1]].par) for class in partition)
    Partition.synchronizeParameters!(ntwk,partition)
    X = Network.evaluate(ntwk,feat)
    λ = Network.adjoint(ntwk, X, label, loss)
    bse = loss.evalutate(label,X,ntwk.results) + penalty.evaluate(λ,ntwk.features,ntwk.hid_dim,ntwk.seq_length, ntwk_type)
    for class in partition
        ntwk_copy = deepcopy(ntwk)
        max_j = maximum(class)
        for j = 1:ntwk.neurons[max_j].par
            ntwk_copy.neurons[max_j].β[j] += ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
            X_copy = Network.evaluate(ntwk_copy,feat)
            λ_copy = Network.adjoint(ntwk_copy, X_copy, label, loss)
            numDer[max_j][j] = (loss.evaluate(label,X_copy,ntwk.results) + penalty.evaluate(λ_copy, ntwk.features,ntwk.hid_dim,ntwk.seq_length, ntwk_type) - bse)/ϵ
            ntwk_copy.neurons[max_j].β[j] -= ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
        end
    end
    return numDer
end

function numerical_parameter_coadjoint(loss :: Module, penalty :: Module, ntwk, partition, feat, label, ntwk_type; ϵ = 1e-8)
    numDer = Dict{Int64,Vector{Float64}}(maximum(class) => zeros(ntwk.neurons[class[1]].par) for class in partition)
    Partition.synchronizeParameters!(ntwk,partition)
    X = Network.evaluate(ntwk,feat)
    λ = Network.adjoint(ntwk, X, label, loss)
    bse = loss.evaluate(label,X,ntwk.results) + penalty.evaluate(λ,ntwk.features,ntwk.hid_dim,ntwk.seq_length, ntwk_type)
    for class in partition
        ntwk_copy = deepcopy(ntwk)
        max_j = maximum(class)
        for j = 1:ntwk.neurons[max_j].par
            ntwk_copy.neurons[max_j].β[j] += ϵ
            Partition.synchronizeParameters!(ntwk_copy,partition)
            X_copy = Network.evaluate(ntwk_copy,feat)
            λ_copy = Network.adjoint(ntwk_copy, X_copy, label, loss)
            numDer[max_j][j] = (loss.evaluate(label,X_copy,ntwk.results) + penalty.evaluate(λ_copy,ntwk.features,ntwk.hid_dim,ntwk.seq_length, ntwk_type) - bse)/ϵ
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
lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
for networkEncap in testNets
    nme, desc, ntwk, partition, feat, label, ntwk_type = networkEncap()

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
    grads = paramGrad(lsmod, pnmod, ntwk, feat, label, syncFlag, partition, ntwk_type)

    printstyled("COMPLETED\n",color=:light_yellow)

    printstyled("Computing (coadjoint) numerical parameter derivative\t")
    gradNum = numerical_parameter_coadjoint(lsmod, pnmod, ntwk, partition, feat, label, ntwk_type)
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
    update!(lsmod, pnmod, copy_ntwk, feat, label, step, syncFlag, partition, ntwk_type)
    maximum([norm(copy_ntwk.neurons[k].β - upd_param[k]) for k in 1:L] ) < 1e-12 ?
        printstyled("PASSED\n\n", color=:green) :
        printstyled("FAILED\n\n", color=:red);
end
