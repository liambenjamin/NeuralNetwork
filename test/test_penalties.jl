include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/RNN.jl")
include("../src/penalties.jl")

using LinearAlgebra, Main.loss, Main.penalty, Main.RNN

loss_list = [:ltwo, :crossEntropy, :binary_crossEntropy, :softmaxCrossEntropy]


printstyled("\n\n**************\nTesting Losses\n**************\n\n",color=:magenta)
for ls in loss_list
    printstyled("Testing: $ls\n", color=:blue)
    #label = [0.01, 0.01, 1.01, 0.01, 0.01]
    label = [0, 0, 1, 0, 0]
    pred = [0.1,0.1,0.6,0.1,0.1]
    if ls == :binary_crossEntropy
        label = 0
        pred = [0.1]
    end
    ϵ = 1e-8
    bse_f = eval(ls).evaluate(label,pred,length(label))
    bse_g = eval(ls).gradient(label,pred,length(label))
    bse_h = eval(ls).hessian(label,pred,length(label))

    # numerical differences
    num_g = zeros(length(label))
    num_h = zeros(length(label), length(label))

    # numerical gradient
    for i=1:length(pred)
        pert = zeros(length(pred))
        pert[i] = ϵ
        num_g[i] = (eval(ls).evaluate(label,pred+pert,length(label)) - bse_f)/ϵ

        num_h[i,:] = (eval(ls).gradient(label,pred+pert,length(label)) - bse_g)/ϵ
    end

    printstyled("\tGradient: ")
    norm(num_g - bse_g) < 1e-6 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red)
    printstyled("$(norm(num_g - bse_g))",color=:blue)

    printstyled("\tHessian: ")
    norm(num_h - bse_h) < 1e-6 ?
        printstyled("PASSED\n\n", color=:green) :
        printstyled("FAILED\n\n", color=:red)
    printstyled("$(norm(num_h - bse_h))",color=:blue)
end


penalty_list = [:log_L2, :log_var, :log_var_phi, :fixed_var]
comp_penalty_list = [:var_phi]

printstyled("\n\n**********************\nTesting Penalty Functions (G):\n**********************\n\n",color=:magenta)
for pk in penalty_list
    printstyled("Testing: $pk\n", color=:blue)

    inp_dim = 25
    hid_dim = 5
    seq_length = 2
    ϵ = 1e-8
    Λ = randn(50)
    bse_f = eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length)
    bse_g = eval(pk).gradient(Λ, inp_dim, hid_dim, seq_length)

    # numerical gradient
    num_g = zeros(50)

    for i = 21:35
        val = Λ[i]
        Λ[i] += ϵ
        num_g[i] = (eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length) - bse_f)/ϵ
        Λ[i] = val
    end
    printstyled("\tGradient: ")
    norm(num_g - bse_g) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red)
    printstyled("Norm Difference: $(norm(num_g - bse_g))\n")

end
printstyled("\n\n**********************\nTesting Composite Penalty Functions (G):\n**********************\n\n",color=:magenta)
for pk in comp_penalty_list
    printstyled("Testing RNN: $pk\n", color=:blue)
    n_type = "rnn"
    inp_dim = 25
    hid_dim = 5
    seq_length = 2
    ϵ = 1e-8
    Λ = randn(35)
    bse_f = eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length, n_type)
    bse_g = eval(pk).gradient(Λ, inp_dim, hid_dim, seq_length, n_type)

    # numerical gradient
    num_g = zeros(size(Λ,1))
    X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    for i in X_pos
        val = Λ[i]
        Λ[i] += ϵ
        num_g[i] = (eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length, n_type) - bse_f)/ϵ
        Λ[i] = val
    end
    printstyled("\tGradient: ")
    norm(num_g - bse_g) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red)
    printstyled("Norm Difference: $(norm(num_g - bse_g))\n")

end

for pk in comp_penalty_list
    printstyled("Testing LSTM: $pk\n", color=:blue)
    n_type = "lstm"
    inp_dim = 30
    hid_dim = 5
    seq_length = 2
    ϵ = 1e-8
    Λ = randn(inp_dim + seq_length*hid_dim*6)
    bse_f = eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length, n_type)
    bse_g = eval(pk).gradient(Λ, inp_dim, hid_dim, seq_length, n_type)

    # numerical gradient
    num_g = zeros(size(Λ,1))
    X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    for i in X_pos
        val = Λ[i]
        Λ[i] += ϵ
        num_g[i] = (eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length, n_type) - bse_f)/ϵ
        Λ[i] = val
    end
    printstyled("\tGradient: ")
    norm(num_g - bse_g) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red)
    printstyled("Norm Difference: $(norm(num_g - bse_g))\n")

end

for pk in comp_penalty_list
    printstyled("Testing GRU: $pk\n", color=:blue)
    n_type = "gru"
    inp_dim = 25
    hid_dim = 5
    seq_length = 2
    ϵ = 1e-8
    Λ = randn(inp_dim + seq_length*hid_dim*5)
    bse_f = eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length, n_type)
    bse_g = eval(pk).gradient(Λ, inp_dim, hid_dim, seq_length, n_type)

    # numerical gradient
    num_g = zeros(size(Λ,1))
    X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    for i in X_pos
        val = Λ[i]
        Λ[i] += ϵ
        num_g[i] = (eval(pk).evaluate(Λ, inp_dim, hid_dim, seq_length, n_type) - bse_f)/ϵ
        Λ[i] = val
    end
    printstyled("\tGradient: ")
    norm(num_g - bse_g) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red)
    printstyled("Norm Difference: $(norm(num_g - bse_g))\n")

end


test_penalty_list = [:test_g]
printstyled("\n\n**********************\nTesting Test Penalty Functions (G):\n**********************\n\n",color=:magenta)
for pk in test_penalty_list
    printstyled("Testing: $pk\n", color=:blue)

    inp_dim = 25
    hid_dim = 5
    seq_length = 2
    output_dim = 2
    ϵ = 1e-8
    Λ = randn(50)
    bse_f = eval(pk).evaluate(Λ, inp_dim, hid_dim, output_dim)
    bse_g = eval(pk).gradient(Λ, inp_dim, hid_dim, output_dim)

    # numerical gradient
    num_g = zeros(50)

    for i = 1:length(Λ)
        val = Λ[i]
        Λ[i] += ϵ
        num_g[i] = (eval(pk).evaluate(Λ, inp_dim, hid_dim, output_dim) - bse_f)/ϵ
        Λ[i] = val
    end
    printstyled("\tGradient: ")
    norm(num_g - bse_g) < 1e-5 ?
        printstyled("PASSED\n", color=:green) :
        printstyled("FAILED\n", color=:red)

end
