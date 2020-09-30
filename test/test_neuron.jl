include("../src/neurons.jl")

using LinearAlgebra, Random, Main.Neurons

one_models = [:bias, :linear, :logistic, :hyptan, :smoothmax, :sigmoid, :hypertan, :lstmCellState, :lstmHiddenState, :gruHiddenState, :hadamardCellState]
multi_models = [:convolution]
sm_models = [:softmax]
params = Dict(
    :bias => [randn(), randn(), Dict(:empty=>0)],
    :linear => [randn(10), randn(11), Dict(:empty=>0)],
    :logistic => [randn(10), randn(12), Dict(:empty=>0)],
    :sigmoid => [randn(10), randn(11), Dict(:empty=>0)],
    :hypertan => [randn(10), randn(11), Dict(:empty=>0)],
    :hyptan => [randn(10), randn(13), Dict(:empty=>0)],
    :smoothmax => [randn(10), rand(), Dict(:empty=>0)],
    :convolution => [randn(50), randn(10), Dict(:met=> [2,3,7])],
    :softmax => [randn(10), randn(), Dict(:empty=>0)],
    :lstmCellState => [randn(4), randn(5), Dict(:empty=>0)],
    :lstmHiddenState => [randn(2), randn(3), Dict(:empty=>0)],
    :gruHiddenState => [randn(3), randn(4), Dict(:empty=>0)],
    :hadamardCellState => [randn(2), randn(3), Dict(:empty=>0)]
    )

printstyled("****************************************\nBeginning unit tests for neurons\n****************************************\n\n"; color=:light_magenta)

for model in one_models
    printstyled("Testing: ",color=:light_magenta); printstyled("$model\n"; bold=true, color=:blue)

    #dX
    print("dX\t")
    ana_der = eval(model).dX(params[model][1:2]...; params[model][3]...)
    num_der = Neurons.differenceDerivative(params[model]..., eval(model).act)
    norm(num_der[:X] - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dP
    print("dP\t")
    ana_der = eval(model).dP(params[model][1:2]...; params[model][3]...)
    norm(num_der[:P] - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dPP
    print("dPP\t")
    ana_der = eval(model).dPP(params[model][1:2]...; params[model][3]...)
    num_der = Neurons.differenceDerivative(params[model]..., eval(model).dP)
    norm(num_der[:P] - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dXX
    print("dXX\t")
    ana_der = eval(model).dXX(params[model][1:2]...; params[model][3]...)
    num_der = Neurons.differenceDerivative(params[model]..., eval(model).dX)
    norm(num_der[:X] - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dXP
    print("dXP\t")
    ana_der = eval(model).dXP(params[model][1:2]...; params[model][3]...)
    norm(num_der[:P] - ana_der) < 1e-6 ?
        printstyled("PASSED\n\n", color=:green) :
        printstyled("FAILED\n\n", color=:red);
end


for model in multi_models
    printstyled("Testing: ",color=:light_magenta); printstyled("$model\n"; bold=true, color=:blue)

    #dX
    print("dX\t")
    ana_der = eval(model).dX(params[model][1:2]...; params[model][3]...)
    num_der = Neurons.differenceDerivative(params[model]..., eval(model).act)
    norm(num_der[:X] - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dP
    print("dP\t")
    ana_der = eval(model).dP(params[model][1:2]...; params[model][3]...)
    norm(num_der[:P] - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dPP
    print("dPP\t")
    ana_der = eval(model).dPP(params[model][1:2]...;params[model][3]...)
    num_der = Neurons.differenceDerivative(params[model]..., eval(model).dP)
    L = length(ana_der)
    sum(map(λ -> norm( ana_der[λ] - num_der[:P][λ,:,:]), 1:L)) < L*1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dXX
    print("dXX\t")
    ana_der = eval(model).dXX(params[model][1:2]...;params[model][3]...)
    num_der = Neurons.differenceDerivative(params[model]..., eval(model).dX)
    L = length(ana_der)
    sum(map(λ -> norm( ana_der[λ] - num_der[:X][λ,:,:]), 1:L)) < L*1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dXP
    print("dXP\t")
    ana_der = eval(model).dXP(params[model][1:2]...;params[model][3]...)
    L = length(ana_der)
    sum(map(λ -> norm( ana_der[λ] - num_der[:P][λ,:,:]), 1:L)) < L*1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);
end


for model in sm_models
    printstyled("Testing: ",color=:light_magenta); printstyled("$model\n"; bold=true, color=:blue)
    #Z = params[model]
    L = length(params[model][1])
    ϵ = 1e-8
    bse = eval(model).act(params[model][1],params[model][2])
    #dX
    print("dX\t")
    ana_der = eval(model).dX(params[model][1],params[model][2])
    num_der = zeros(L,L)
    for i=1:L
        pert = zeros(L)
        pert[i] = ϵ
        num_der[i,:] = (eval(model).act(params[model][1]+pert,params[model][2]) - bse) ./ ϵ
    end
    norm(num_der - ana_der) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

    #dXX
    print("dXX\t")
    bse_g = eval(model).dX(params[model][1],params[model][2])
    ana_der = eval(model).dXX(params[model][1],params[model][2])
    num_der = zeros(L,L,L)
    for i=1:L
        pert = zeros(L)
        pert[i] = ϵ
        num_der[i,:,:] = (eval(model).dX(params[model][1]+pert,params[model][2]) - bse_g) ./ ϵ
    end

    #L = length(ana_der)
    sum(map(λ -> norm( ana_der[λ,:,:] - num_der[λ,:,:]), 1:L)) < L*1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red);

end
