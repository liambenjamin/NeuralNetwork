#Author: Liam Johnston
#Date: June 3, 2020
#Description: Functions used during training of neural networks
include("rnn.jl")
include("lstm.jl")
include("gru.jl")

module loss

# Note 1: `label` is true label (vector or scalar)
# Note 2: `X` where `X[end-output_dim+1:end]` is the network's prediction

export ltwo, crossEntropy, softmaxCrossEntropy

module ltwo
using LinearAlgebra

evaluate(label,X,output_dim) = 0.5 * dot(label-X[end-output_dim+1:end], label - X[end-output_dim+1:end])
gradient(label,X,output_dim) = X[end-output_dim+1:end] - label

function hessian(label,X,output_dim)
    outInd = length(X)-length(label)+1:length(X)
    J = diagm(0 => zeros(length(X)))
    J[outInd,outInd] = diagm(0 => ones(length(label)))
    return J
end

end # end ltwo

module crossEntropy
using LinearAlgebra

evaluate(label,X,output_dim) = -label' * log.(X[end-output_dim+1:end])

function gradient(label,X,output_dim)
    outInd = length(X)-length(label)+1:length(X)
    ∂f = zeros(length(X))
    ∂f[outInd] = -label .* X[outInd].^-1
    return ∂f
end

function hessian(label,X,output_dim)
    outInd = length(X)-length(label)+1:length(X)
    ∂f2 = zeros(length(X), length(X))
    ∂f2[outInd, outInd] = diagm(0 => label .* X[outInd].^-2)
    return ∂f2
end
end # end crossEntropy

module softmaxCrossEntropy
using LinearAlgebra, Statistics

"""
Implements softmax crossentropy function, gradient and hessian.
    - label: y or true label
    - X: network architecture vector
    - output_dim: dimension of output
"""

function evaluate(label, X, output_dim)
    i = findall(x -> x == 1, label)[]
    z = X[end-output_dim+1:end]
    ls = log(sum(exp.(z)))
    return ls - z[i]
end

function gradient(label, X, output_dim)
    i = findall(x -> x == 1, label)[]
    δ = zeros(output_dim)
    δ[i] = 1
    m = maximum(X[end-output_dim+1:end])
    ∂Z = exp.(X[end-output_dim+1:end] .- m) / (sum(exp.(X[end-output_dim+1:end] .- m))) .- δ

    # must return vector of length(X)
    #ind = length(X)-length(label)+1:length(X)
    ∂f = zeros(length(X))
    ∂f[length(X)-length(label)+1:length(X)] = ∂Z
    return ∂f
end

function hessian(label,X,output_dim)
    z = X[end-output_dim+1:end]
    S = sum(exp.(z))
    J = reshape(collect(Iterators.flatten(map(i -> -exp.(z[i] .+ z) / S^2, 1:output_dim))), output_dim, output_dim)
    J[diagind(J)] += exp.(z) / S


    # returns a matrix of dimension: (length(X), length(X))
    outInd = length(X)-length(label)+1:length(X)
    ∂f2 = zeros(length(X), length(X))
    ∂f2[outInd, outInd] = J
    return ∂f2
end
end # end softmaxCrossEntropy



end # end loss module

####################################################

module penalty

export var_phi, log_L2, log_var, log_var_phi, fixed_var, test_g

module test_g
using LinearAlgebra

# general penalty function
function evaluate(L, inp_dim, hid_dim, output_dim)
    λ = L[1:end-output_dim]
    return sum(1.0 ./ norm.(λ)) / 1e5
end

function gradient(L, inp_dim, hid_dim, output_dim)
    ∂g = zeros(length(L))
    λ = L[1:end-output_dim]
    ∂g[1:end-output_dim] = -(1.0 ./ norm.(λ).^2) .* ( λ ./ norm.(λ) )
    return ∂g ./ 1e5
end

end # end test_g module


module var_phi
using LinearAlgebra, Main.RNN, Main.LSTM, Main.GRU

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64, ntwk_type :: String)
    if ntwk_type == "rnn"
        X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    elseif ntwk_type == "lstm"
        X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    else
        X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    end

    Λ = reshape(L[X_pos], (hid_dim, seq_length+1))

    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    ϕ = map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ1
        -norm(λ_star) / (norm(Λ[:,T][i])*(1+log(norm(λ_star)^2))) + 1 :
        # ϕ2
        log(norm(λ_star)*norm(Λ[:,T][i])) / (1+log(norm(λ_star)^2)), 1:d)
    return 1e-2 * ((sum( nrms.^2)/T - M^2) + 20 * sum(ϕ))
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64, ntwk_type :: String)
    if ntwk_type == "rnn"
        X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    elseif ntwk_type == "lstm"
        X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    else
        X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    end

    Λ = reshape(L[X_pos], (hid_dim, seq_length+1))

    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    val = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : ((2/T)*Λ[:,i] - (2*M/T)* Λ[:,i]/nrms[i]) , 1:T)...)
    ∂ϕ = nrms[T] == 0 ? zeros(d) : map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ'1
        (norm(λ_star)*Λ[:,T][i]) / (norm(Λ[:,T][i])^3 * (1+log(norm(λ_star)^2))) :
        # ϕ'2
        (norm(λ_star)*Λ[:,T][i]) / (norm(λ_star) * (1+log(norm(λ_star)^2)) * norm(Λ[:,T][i])^2), 1:d)
    val[:,T] = nrms[T] == 0 ? zeros(d) : val[:,T] + 20 * ∂ϕ
    # append vector to adjoint vector
    grad = zeros(length(L))
    grad[X_pos] = collect(Iterators.flatten(val))
    return 1e-2 * grad
end

end #end var_phi module

module urban_var_phi
using LinearAlgebra, Main.RNN, Main.LSTM, Main.GRU

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64, ntwk_type :: String)
    if ntwk_type == "rnn"
        X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    elseif ntwk_type == "lstm"
        X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    else
        X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    end

    Λ = reshape(L[X_pos], (hid_dim, seq_length+1))

    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    ϕ = map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ1
        -norm(λ_star) / (norm(Λ[:,T][i])*(1+log(norm(λ_star)^2))) + 1 :
        # ϕ2
        log(norm(λ_star)*norm(Λ[:,T][i])) / (1+log(norm(λ_star)^2)), 1:d)
    return 1e-1 * ((sum( nrms.^2)/T - M^2) + 20 * sum(ϕ))
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64, ntwk_type :: String)
    if ntwk_type == "rnn"
        X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    elseif ntwk_type == "lstm"
        X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    else
        X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    end

    Λ = reshape(L[X_pos], (hid_dim, seq_length+1))

    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    val = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : ((2/T)*Λ[:,i] - (2*M/T)* Λ[:,i]/nrms[i]) , 1:T)...)
    ∂ϕ = nrms[T] == 0 ? zeros(d) : map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ'1
        (norm(λ_star)*Λ[:,T][i]) / (norm(Λ[:,T][i])^3 * (1+log(norm(λ_star)^2))) :
        # ϕ'2
        (norm(λ_star)*Λ[:,T][i]) / (norm(λ_star) * (1+log(norm(λ_star)^2)) * norm(Λ[:,T][i])^2), 1:d)
    val[:,T] = nrms[T] == 0 ? zeros(d) : val[:,T] + 20 * ∂ϕ
    # append vector to adjoint vector
    grad = zeros(length(L))
    grad[X_pos] = collect(Iterators.flatten(val))
    return 1e-1 * grad
end

end #end urban_var_phi module


module var_phi_lstm
using LinearAlgebra, Main.RNN, Main.LSTM, Main.GRU

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64, ntwk_type :: String)
    if ntwk_type == "rnn"
        X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    elseif ntwk_type == "lstm"
        X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    else
        X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    end

    Λ = reshape(L[X_pos], (hid_dim, seq_length+1))

    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    ϕ = map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ1
        -norm(λ_star) / (norm(Λ[:,T][i])*(1+log(norm(λ_star)^2))) + 1 :
        # ϕ2
        log(norm(λ_star)*norm(Λ[:,T][i])) / (1+log(norm(λ_star)^2)), 1:d)
    return 1e-1 * ((sum( nrms.^2)/T - M^2) + sum(ϕ))
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64, ntwk_type :: String)
    if ntwk_type == "rnn"
        X_pos = RNN.rnnXind(inp_dim-hid_dim,hid_dim,seq_length)
    elseif ntwk_type == "lstm"
        X_pos = LSTM.lstmXind(inp_dim-2*hid_dim,hid_dim,seq_length)
    else
        X_pos = GRU.gruXind(inp_dim-hid_dim,hid_dim,seq_length)
    end

    Λ = reshape(L[X_pos], (hid_dim, seq_length+1))

    d, T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    val = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : ((2/T)*Λ[:,i] - (2*M/T)* Λ[:,i]/nrms[i]) , 1:T)...)
    ∂ϕ = nrms[T] == 0 ? zeros(d) : map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ'1
        (norm(λ_star)*Λ[:,T][i]) / (norm(Λ[:,T][i])^3 * (1+log(norm(λ_star)^2))) :
        # ϕ'2
        (norm(λ_star)*Λ[:,T][i]) / (norm(λ_star) * (1+log(norm(λ_star)^2)) * norm(Λ[:,T][i])^2), 1:d)
    val[:,T] = nrms[T] == 0 ? zeros(d) : val[:,T] + ∂ϕ
    # append vector to adjoint vector
    grad = zeros(length(L))
    grad[X_pos] = collect(Iterators.flatten(val))
    return 1e-1 * grad
end

end #end var_phi_lstm module


module log_L2
using LinearAlgebra

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    layer_T = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[layer_T], hid_dim, seq_length+1)
    d, T = size(Λ)
    log_nrms = map(i -> norm(Λ[:,i]) == 0.0 ? 0.0 : log(norm(Λ[:,i])^2), 1:T)
    return sum( map(i -> log_nrms[i] == 0.0 ? 0.0 : -1/2 * log_nrms[i], 1:T) )
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    layer_T = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[layer_T], hid_dim, seq_length+1)
    d,T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    dg_mat = hcat(map(i ->  nrms[i] == 0 ? 0.0 : (-Λ[:,i]) ./ (norm(Λ[:,i])^2) , 1:T)...)
    dg = zeros(length(L))
    dg[layer_T] = collect(Iterators.flatten(dg_mat))
    return dg
end

end # end log_L2 module

module log_var
using LinearAlgebra

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    # reshape input L to matrix sorted by layer
    layer_T = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[layer_T], hid_dim, seq_length+1)
    d, T = size(Λ)

    # log penalty
    log_nrms = map(i -> norm(Λ[:,i]) == 0.0 ? 0.0 : log(norm(Λ[:,i])^2), 1:T)
    pnLog = sum( map(i -> log_nrms[i] == 0.0 ? 0.0 : -1/2 * log_nrms[i], 1:T) )

    # var penalty
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    pnVar = sum( nrms.^2)/T - M^2
    return pnLog + pnVar
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    # reshape input L to matrix sorted by layer
    layer_T = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[layer_T], hid_dim, seq_length+1)
    d,T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    # ∂log
    ∂log_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : (-Λ[:,i]) ./ (norm(Λ[:,i])^2) , 1:T)...)
    ∂log = collect(Iterators.flatten(∂log_mat))
    # ∂var
    M = sum(nrms)/T
    ∂var_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : ((2/T)*Λ[:,i] - (2*M/T)* Λ[:,i]/nrms[i]) , 1:T)...)
    ∂var = collect(Iterators.flatten(∂var_mat))
    # append vector to adjoint vector
    grad = zeros(length(L))
    grad[layer_T] = ∂log + ∂var
    return grad
end

end #end log_var module

module log_var_phi
using LinearAlgebra

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    # reshape input L to matrix sorted by layer
    layer_T = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[layer_T], hid_dim, seq_length+1)
    d, T = size(Λ)

    # log penalty
    log_nrms = map(i -> log(norm(Λ[:,i])^2), 1:T)
    pnLog = sum( map(i -> log_nrms[i] == 0.0 ? 0.0 : -1/2 * log_nrms[i], 1:T) )

    # var penalty
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    M = sum(nrms)/T
    λ_star = 1e-5
    ϕ = map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ1
        -norm(λ_star) / (norm(Λ[:,T][i])*(1+log(norm(λ_star)^2))) + 1 :
        # ϕ2
        log(norm(λ_star)*norm(Λ[:,T][i])) / (1+log(norm(λ_star)^2)), 1:d)
    pnVar = sum( nrms.^2)/T - M^2 + sum(ϕ)
    return pnLog + pnVar
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    # reshape input L to matrix sorted by layer
    layer_T = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[layer_T], hid_dim, seq_length+1)
    d,T = size(Λ)
    nrms = map( i -> norm(Λ[:,i]), 1:T)
    # ∂log
    ∂log_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : (-Λ[:,i]) ./ (norm(Λ[:,i])^2) , 1:T)...)
    ∂log = collect(Iterators.flatten(∂log_mat))
    # ∂var_phi
    M = sum(nrms)/T
    ∂var_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : ((2/T)*Λ[:,i] - (2*M/T)* Λ[:,i]/nrms[i]) , 1:T)...)
    λ_star = 1e-5
    ∂ϕ = nrms[T] == 0 ? zeros(d) : map( i -> norm(Λ[:,T][i]) < norm(λ_star) ?
        # ϕ'1
        (norm(λ_star)*Λ[:,T][i]) / (norm(Λ[:,T][i])^3 * (1+log(norm(λ_star)^2))) :
        # ϕ'2
        (norm(λ_star)*Λ[:,T][i]) / (norm(λ_star) * (1+log(norm(λ_star)^2)) * norm(Λ[:,T][i])^2), 1:d)
    ∂var_mat[:,T] = nrms[T] == 0 ? zeros(d) : ∂var_mat[:,T] + ∂ϕ
    ∂var = collect(Iterators.flatten(∂var_mat))

    # append vector to adjoint vector
    grad = zeros(length(L))
    grad[layer_T] = ∂log + ∂var
    return grad
end

end #end log_var_phi module

module fixed_var
using LinearAlgebra

function evaluate(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    act_nrn = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)#785:813
    Λ = reshape(L[act_nrn], hid_dim, seq_length+1)
    _, T = size(Λ)
    c = 1e-4
    fixed_var = map(i -> (norm(Λ[:,i]) - c)^2, 1:T)
    # log penalty
    log_nrms = map(i -> log(norm(Λ[:,i])^2), 1:T)
    pnLog = sum( map(i -> log_nrms[i] == 0.0 ? 0.0 : -1/2 * log_nrms[i], 1:T) )
    return sum(fixed_var) / T + pnLog
end

function gradient(L :: Vector, inp_dim :: Int64, hid_dim :: Int64, seq_length :: Int64)
    act_nrn = inp_dim-hid_dim+1:(inp_dim + hid_dim*seq_length)
    Λ = reshape(L[act_nrn], hid_dim, seq_length+1)
    d, T = size(Λ)
    nrms = map(i -> norm(Λ[:,i]), 1:T)
    c = 1e-4
    # ∂log
    ∂log_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : (-Λ[:,i]) ./ (norm(Λ[:,i])^2) , 1:T)...)
    ∂log = collect(Iterators.flatten(∂log_mat))
    # ∂var_fixed
    grad_mat = hcat(map(i ->  nrms[i] == 0 ? zeros(d) : (2/T) * (norm(Λ[:,i])-c) * (Λ[:,i] ./ norm(Λ[:,i])), 1:T)...)
    ∂g = collect(Iterators.flatten(grad_mat))
    grad = zeros(length(L))
    grad[act_nrn] = ∂g + ∂log

    return grad
end

end #end fixed_var


end # end penalty module
