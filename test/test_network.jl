include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/lstm.jl")
include("../src/gru.jl")
include("../src/penalties.jl")


using LinearAlgebra, SparseArrays, Main.Neurons, Main.Network, Main.loss, Main.penalty

#Experimental Networks
testNets = Function[
function makeSimpleLogistic()

    #Assumes 3 dimensional input
    #Assumes each input passes to logistic neuron
    #Outputs 1 dimensional quantity from logistic neuron

    inp_dim, hid_dim, fc_dim, seq_length, output_dim, label = 6, 3, 1, 0, 3, [1.0]

    neurons = Vector{neuron}(undef,1)
    neurons[1] = logistic.init(1,3,5)

    rowInd = [1,2,3,4]
    colInd = [1,1,1,2]
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd,colInd,vals)

    ntwk = Network.network(neurons,3,1, hid_dim, fc_dim, seq_length)
    Network.graph!(ntwk,graph)

    #Name
    name = "Simple Logistic Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 3-Dimen
    \tFull Co.\t |||||
    \tNeuron  \t Logistic
    \tFull Co.\t |||||
    \tOutput  \t 1-Dimen \n\n
    """

    return name, desc, ntwk, 0.01*randn(3), label
end#=,

function makeSimpleSigmoidSoftmax()

    #Assumes 3 dimensional input
    #Assumes each input passes to sigmoid neuron (3 in total)
    #Assumes each logistic neuron feeds into a (single) softmax neuron
    #Outputs 3 dimensional quantity from softmax neuron

    inp_dim, hid_dim, fc_dim, seq_length, output_dim= 6, 3, 0, 1, 3
    label = zeros(output_dim)
    label[2] = 1.0

    neurons = Vector{neuron}(undef,4)
    neurons[1] = sigmoid.init(1,3,4)
    neurons[2] = sigmoid.init(2,3,4)
    neurons[3] = sigmoid.init(3,3,4)
    neurons[4] = softmax.init(4,3,1)

    rowInd = vcat(1:3, 1:3, 1:3, 4:6, 7:9)
    colInd = vcat(ones(Int64, 3), 2*ones(Int64,3), 3*ones(Int64,3), 4*ones(Int64,3), 5*ones(Int64,3))

    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd,colInd,vals)


    ntwk = Network.network(neurons,3,3,hid_dim, fc_dim, seq_length) # (neuron list, input dim, output dim)
    Network.graph!(ntwk,graph)

    #Name
    name = "Simple Logistic Softmax Output Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 3-Dimen
    \tFull Co.\t |||||
    \tNeuron  \t Sigmoid
    \tFull Co.\t |||||
    \tOutput  \t 3-Dimen \n\n
    """

    return name, desc, ntwk, 0.01*randn(3), label
end,
function makeTwoLayerLinearLogistic()

    #Assumes 16 inputs
    #3 linear neurons taking inputs 1-8, 5-12, 9-16
    #3 logistic neurons full connected to three linear neurons
    #Outputs 3 values passed through softmax neuron

    inp_dim, hid_dim, fc_dim, seq_length, output_dim= 16, 3, 0, 2, 3
    label = zeros(output_dim)
    label[2] = 1.0

    neurons = Vector{neuron}(undef,7)
    neurons[1] = linear.init(1,8,9)
    neurons[2] = linear.init(2,8,9)
    neurons[3] = linear.init(3,8,9)
    neurons[4] = logistic.init(4,3,5)
    neurons[5] = logistic.init(5,3,5)
    neurons[6] = logistic.init(6,3,5)
    neurons[7] = softmax.init(7,3,1)

    rowInd = vcat(1:8, 5:12, 9:16, 17:19, 17:19, 17:19, 20:22, 23:25)
    colInd = vcat(ones(Int64, 8), 2*ones(Int64, 8), 3*ones(Int64, 8), 4*ones(Int64, 3), 5*ones(Int64, 3), 6*ones(Int64, 3), 7*ones(Int64, 3), 8*ones(Int64,3) )
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd, colInd, vals)

    ntwk = Network.network(neurons, 16, 3, hid_dim, fc_dim, seq_length)
    Network.graph!(ntwk, graph)

    #Name
    name = "Two-layer Linear-Logistic Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 16-Dimen
    \tPart Co.\t |  |  |
    \tLayer   \t 3 Logistic
    \tFull Co.\t |||||||
    \tLayer   \t 3 Linear
    \tPart Co.\t |  |  |
    \tOutput  \t 3-Dimen \n\n
    """

    return name, desc, ntwk, 0.01*randn(16), label
end,
function makeTwoLayerLogisticLogistic()

    #Assumes 2 dimensional input
    #Both inputs feed into 2 logistic neurons
    #Outputs from both logistic neurons feed into 2 more logistic neurons
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

    label = zeros(2)
    label[1] = 1.0

    ntwk = Network.network(neurons,2,2,2,0,2)
    Network.graph!(ntwk,graph)

    #Name
    name = "Two-layer Logistic-Logistic Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 2-Dimen
    \tFull Co \t |||||||
    \tLayer   \t 2 Logistic
    \tFull Co \t |||||||
    \tLayer   \t 2 Logistic
    \tPart Co \t |  |  |
    \tOutput  \t 2-Dimen\n\n
    """

    return name, desc, ntwk, 0.01*randn(2), label
end,
function makeConvolutionSmoothMax()

    #Assumes 10 dimensional input
    #Inputs feed into a layer of 2 convolution neurons
    #Layer outputs feed into a single smoothmax function
    #Outputs 1 dimension value from smoothmax function

    neurons = Vector{neuron}(undef,3)
    neurons[1] = convolution.init(1,10,2,met=[3,0,3])
    neurons[2] = convolution.init(2,10,3,met=[1,0,4])
    neurons[3] = smoothmax.init(3,6,1)

    rowInd = vcat(1:10, 1:10, 11:16,17)
    colInd = vcat(1*ones(Int64,10), 2*ones(Int64,10), 3*ones(Int64,6), 4)
    vals = ones(Int64, length(rowInd))
    graph = sparse(rowInd, colInd, vals)

    ntwk = Network.network(neurons,10,1,2,0,2) #Neurons, Input dimenion, Output dimension
    Network.graph!(ntwk,graph)

    #Name
    name = "Two-layer Convolution-SmoothMax Network"

    #Description
    desc = """
    \tNetwork Description

    \tFeature \t 10-Dimen
    \tFull Co \t |||||||
    \tLayer   \t 2 1-d-Conv
    \tFull Co \t |||||||
    \tNeuron  \t Softmax
    \tFull Co \t |||||||
    \tOutput  \t 1-Dimen\n\n
    """

    return name, desc, ntwk, 0.01*randn(10)
end
=#
]

#Numerical Derivatives
function numerical_adjoint(loss :: Module, ntwk, feat, label; ϵ = 1e-8)
    t = length(feat)
    E = diagm(0 => ϵ*ones(t))
    bse = loss.evaluate(label, Network.evaluate(ntwk, feat), ntwk.results)
    pert = collect(Iterators.flatten(map(z -> loss.evaluate(label, Network.evaluate(ntwk, feat+ E[:,z]), ntwk.results), 1:t)))
    return (bse .- pert)./ϵ #derivative is the negative adjoint
end

function numerical_parameter_adjoint(loss :: Module, ntwk, feat, label; ϵ = 1e-8)
    L = length(ntwk.neurons)
    numDer = Dict{Int64,Vector{Float64}}(k => zeros(ntwk.neurons[k].par) for k=1:L)
    X = Network.evaluate(ntwk,feat)
    f_bse = loss.evaluate(label,X,ntwk.results)
    for i = 1:L
        ntwk_copy = deepcopy(ntwk)
        for j = 1:ntwk.neurons[i].par
            ntwk_copy.neurons[i].β[j] += ϵ
            X_copy = Network.evaluate(ntwk_copy,feat)
            numDer[i][j] = ( loss.evaluate(label,X_copy,ntwk.results) - f_bse)/ϵ
            ntwk_copy.neurons[i].β[j] -= ϵ
        end
    end
    return numDer
end

function numerical_back_coadjoint(loss :: Module, penalty :: Module, ntwk, feat, label; ϵ = 1e-8 )
    t = length(feat)
    E = diagm(0 => ϵ*ones(t))
    X = Network.evaluate(ntwk,feat)
    λ = Network.adjoint(ntwk, X, label, loss)
    #pnParam = penalty == Main.penalty.test_g ? ntwk.results : ntwk.seq_length
    bse = loss.evaluate(label,X,ntwk.results) + penalty.evaluate(λ,ntwk.features, ntwk.hid_dim, ntwk.seq_length)
    num_coadj = zeros(t)
    for i = 1:t
        X_ϵ = Network.evaluate(ntwk, feat+ E[:,i])
        λ_ϵ = Network.adjoint(ntwk, X_ϵ, label, loss)
        pert = loss.evaluate(label,X_ϵ,ntwk.results) + penalty.evaluate(λ_ϵ,ntwk.features, ntwk.hid_dim, ntwk.seq_length)
        num_coadj[i] = (bse - pert)/ϵ #derivative is negative of coadjoint
    end

    return num_coadj
end

function numerical_parameter_coadjoint(loss :: Module, penalty :: Module, ntwk, feat, label; ϵ = 1e-8)
    L = length(ntwk.neurons)
    numDer = Dict{Int64,Vector{Float64}}(k => zeros(ntwk.neurons[k].par) for k=1:L)
    X = Network.evaluate(ntwk,feat)
    λ = Network.adjoint(ntwk, X, label, loss)
    bse = loss.evaluate(label,X,ntwk.results) + penalty.evaluate(λ,ntwk.features, ntwk.hid_dim, ntwk.seq_length)
    for i = 1:L
        ntwk_copy = deepcopy(ntwk)
        for j = 1:ntwk.neurons[i].par
            ntwk_copy.neurons[i].β[j] += ϵ
            X_copy = Network.evaluate(ntwk_copy,feat)
            λ_copy = Network.adjoint(ntwk_copy, X_copy, label, loss)
            numDer[i][j] = ( loss.evaluate(label,X_copy,ntwk.results) + penalty.evaluate(λ_copy,ntwk.features, ntwk.hid_dim, ntwk.seq_length) - bse)/ϵ
            ntwk_copy.neurons[i].β[j] -= ϵ
        end
    end
    return numDer
end


printstyled("****************************************\nBeginning unit tests for networks\n****************************************\n\n"; color=:light_magenta)

for networkEncap in testNets

    nme, desc, ntwk, feat, label = networkEncap()
    label = label .+ 0.01

    lsmod = loss.crossEntropy
    pnmod = penalty.test_g

    printstyled("Testing: ",color=:light_magenta); printstyled("$nme\n\n"; bold=true, color=:blue)
    printstyled(desc, color=:blue)

    #Evaluate Network
    printstyled("Evaluating network\t\t\t\t\t")
    X = Network.evaluate(ntwk,feat)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Evaluate Adjoint
    printstyled("Evaluating adjoint\t\t\t\t\t")
    λ = Network.adjoint(ntwk, X, label, lsmod)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Compute numerical differences adjoint
    printstyled("Computing numerical adjoint\t\t\t\t")
    λ_0_ϵ = numerical_adjoint(lsmod, ntwk, feat, label)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Comparing Numerical Adjoint and Adjoint
    printstyled("Testing adjoint\t\t\t\t\t\t")
    t = length(feat)
    norm(λ_0_ϵ - λ[1:t]) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Evaluate Parameter Gradient Adjoint
    printstyled("Computing (adjoint) parameter derivative\t\t")
    dB = Network.paramGrad(ntwk, X, λ)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Compute Numerical differences
    printstyled("Computing (adjoint-numerical) parameter derivative\t")
    dBNum = numerical_parameter_adjoint(lsmod, ntwk, feat, label)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Compare Numerical Parameter Derivative and Actual Difference
    printstyled("Comparing (adjoint) parameter derivative\t\t")
    maximum([ norm(dB[k] - dBNum[k] ) for k in keys(dB)]) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Compare adjoint-parameter derivative wrapper
    printstyled("Comparing (adjoint-wrapper) parameter derivative\t")
    dBwrap = Network.paramGrad(lsmod, ntwk, feat, label)
    maximum([ norm(dB[k] - dBwrap[k] ) for k in keys(dB)]) < 1e-12 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Evaluate update function for adjoint
    printstyled("Testing (adjoint) update function\t\t\t")
    L = length(ntwk.neurons)
    step = 0.1
    copy_ntwk = deepcopy(ntwk)
    old_param = Dict(i => copy_ntwk.neurons[i].β for i=1:L)
    upd_param = Dict(i => copy_ntwk.neurons[i].β - step*dB[i] for i = 1:L)
    update!(lsmod, copy_ntwk, feat, label, step)
    maximum([norm(copy_ntwk.neurons[k].β - upd_param[k]) for k in 1:L]) < 1e-12 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Evaluate Forward Coadjoint
    printstyled("Evaluating forward coadjoint\t\t\t\t")
    γ = Network.coadjoint_forward(ntwk,X, λ, pnmod)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Evaluate Backward Coadjoint
    printstyled("Evaluating backward coadjoint\t\t\t\t")
    α = Network.coadjoint_backward(ntwk, X, label, λ, γ, lsmod)
    printstyled("COMPLETED\n",color=:light_yellow)

    #compute numerical differences backward coadjoint
    printstyled("Computing numerical backward coadjoint\t\t\t")
    α_0_ϵ = numerical_back_coadjoint(lsmod, pnmod, ntwk, feat, label)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Comparing Numerical coadjoint and coadjoint
    printstyled("Comparing backward coadjoints\t\t\t\t")
    t = length(feat)
    norm(α_0_ϵ - α[1:t]) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Evaluate Parameter Gradient Adjoint
    printstyled("Computing (coadjoint) parameter derivative\t\t")
    dB = Network.paramGrad(ntwk, X, λ, γ, α)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Compute Numerical differences
    printstyled("Computing (coadjoint-numerical) parameter derivative\t")
    dBNum = numerical_parameter_coadjoint(lsmod, pnmod, ntwk, feat, label)
    printstyled("COMPLETED\n",color=:light_yellow)

    #Compare Numerical Parameter Derivative and Actual Difference
    printstyled("Comparing (coadjoint) parameter derivative\t\t")
    maximum([ norm(dB[k] - dBNum[k] ) for k in keys(dB)]) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Compare adjoint-parameter derivative wrapper
    printstyled("Comparing (coadjoint-wrapper) parameter derivative\t")
    dBwrap = Network.paramGrad(lsmod, pnmod, ntwk, feat, label)
    maximum([ norm(dB[k] - dBwrap[k] ) for k in keys(dB)]) < 1e-12 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #Evaluate update function for adjoint
    printstyled("Testing (coadjoint) update function\t\t\t")
    L = length(ntwk.neurons)
    step = 0.1
    copy_ntwk = deepcopy(ntwk)
    old_param = Dict(i => copy_ntwk.neurons[i].β for i=1:L)
    upd_param = Dict(i => copy_ntwk.neurons[i].β - step*dB[i] for i = 1:L)
    update!(lsmod, pnmod, copy_ntwk, feat, label, step)
    maximum([norm(copy_ntwk.neurons[k].β - upd_param[k]) for k in 1:L]) < 1e-12 ?
        printstyled("PASSED\n\n", color=:green) :
        printstyled("FAILED\n\n", color=:red);
end
