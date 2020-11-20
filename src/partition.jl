#Tools for verification and synchronization across parameters with (possible) equality partitions

module Partition

using LinearAlgebra, SparseArrays, Main.Neurons, Main.Network

import Main.Network: paramGrad, inputGrad, update!

export paramGrad, update!, adam_update!, line_search!

"""
Generate default maximal partition. That is, each parameter is independent.
"""
function defaultPartition(ntwk)
    L = length(ntwk.neurons)
    return map(k -> [k], 1:L)
end

"""
Verifies partition over neurons. Returns synchronization flag.
"""
function verifyPartition(ntwk, partition)
    L = length(ntwk.neurons)
    if (sum(length.(partition)) == L && length(partition) == L)
        return false
    elseif sum(length.(partition)) == L
        for class in partition
            #Number of elements in class
            n = length(class)

            #Verify parameters in a single class are same dimension
            z = map(l -> ntwk.neurons[l].par, class)
            sum(z[1] .== z) == n || @error "Elements in class $class have differing parameter dimensions."
        end

        return true
    else
        @error "Partition does not take an allowed value.\npartition = $partition"
    end
end

"""
Synchronize parameters over partition (verified) to parameter with largest index.
"""
function synchronizeParameters!(ntwk, partition)
    for class in partition
        if length(class) == 1
            continue
        else
            j_star = maximum(class)
            parameter = ntwk.neurons[j_star].β
            for indx in class
                ntwk.neurons[indx].β = parameter
            end
        end
    end
end

"""
Synchronize gradients over partition (verified). Returns gradient corresponding to largest index in class.
"""
function synchronizeGradient(gradient :: Dict{Int64,Vector{Float64}}, partition)
    syncGrad = Dict{Int64,Vector{Float64}}()
    for class in partition
        syncGrad[maximum(class)] = sum( map(λ -> gradient[λ], class))
    end
    return syncGrad
end

"""
Extends paramGrad from network.jl with synchronization
"""
function paramGrad(df :: Function, ntwk :: Network.network, feat :: Vector{Float64}, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    grads = paramGrad(df, ntwk, feat)
    return syncFlag ? synchronizeGradient(grads, partition) : grads
end
function paramGrad(loss :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, syncFlag :: Bool, partition :: Vector{Vector{Int64}} )
    grads = paramGrad(loss, ntwk, feat, label)
    return syncFlag ? synchronizeGradient(grads, partition) : grads
end
function paramGrad(df :: Function, d2f :: Function, dg :: Function, ntwk :: Network.network, feat :: Vector{Float64}, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    grads = paramGrad(df, d2f, dg, ntwk, feat)
    return syncFlag ? synchronizeGradient(grads, partition) : grads
end
function paramGrad(loss :: Module, penalty :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    grads = paramGrad(loss, penalty, ntwk, feat, label)
    return syncFlag ? synchronizeGradient(grads, partition) : grads
end
function paramGrad(loss :: Module, penalty :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String)
    grads = paramGrad(loss, penalty, ntwk, feat, label, ntwk_type)
    return syncFlag ? synchronizeGradient(grads, partition) : grads
end

"""
Extends update! from network.jl with synchronization
"""
function update!(df :: Function, ntwk :: Network.network, feat :: Vector{Float64}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    if syncFlag
        L = length(ntwk.neurons)
        grads = paramGrad(df, ntwk, feat, syncFlag, partition)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(df, ntwk, feat, step)
    end
end
function update!(loss :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    if syncFlag
        L = length(ntwk.neurons)
        grads = paramGrad(loss, ntwk, feat, label, syncFlag, partition)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(loss, ntwk, feat, step)
    end
end
function update!(df :: Function,d2f :: Function, dg :: Function, ntwk :: Network.network, feat :: Vector{Float64}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    if syncFlag
        L = length(ntwk.neurons)
        grads = paramGrad(df, d2f, dg, ntwk, feat, syncFlag, partition)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(df, d2f, dg, ntwk, feat)
    end
end
function update!(loss :: Module, penalty :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    if syncFlag
        L = length(ntwk.neurons)
        grads = paramGrad(loss, penalty, ntwk, feat, label, syncFlag, partition)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(loss, penalty, ntwk, feat)
    end
end
function update!(loss :: Module, penalty :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String)
    if syncFlag
        L = length(ntwk.neurons)
        grads = paramGrad(loss, penalty, ntwk, feat, label, syncFlag, partition, ntwk_type)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(loss, penalty, ntwk, feat)
    end
end

function update!(ntwk :: Network.network, grads :: Vector{Dict{}}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    if syncFlag
        L = length(ntwk.neurons)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(ntwk, grads, step)
    end
end

function update!(ntwk :: Network.network, grads :: Vector{Dict{}}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    if syncFlag
        L = length(ntwk.neurons)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)
    else
        update!(ntwk, grads, step)
    end
end

function adam_update!(ntwk :: Network.network, grads :: Dict{Int64,Vector{Float64}}, step :: Float64, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, m, v, iter)
    if syncFlag
        β1 = 0.9
        β2 = 0.999
        ϵ = 1e-7
        L = length(ntwk.neurons)
        ind = 1
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            # Adam Update
            m[ind] = β1 * m[ind] + (1 - β1) * grads[max_j]
            v[ind] = β2 * v[ind] + (1 - β2) * grads[max_j].^2
            m_hat = m[ind] / (1 - β1^iter)
            v_hat = v[ind] / (1 - β2^iter)

            # Update parameters of maximum index in the class
            ntwk.neurons[max_j].β -= step * (m_hat ./ (sqrt.(v_hat) .+ ϵ))

            # Update ind
            ind += 1
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)

    else
        update!(ntwk, grads, step)
    end
end


"""
Returns a copied network with δ=P-α∇dP parameters
"""

function delta_network(ntwk :: Network.network, step :: Float64, grads :: Dict, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
    δ_ntwk = deepcopy(ntwk)

    if syncFlag
        L = length(δ_ntwk.neurons)
        for class in partition
            #Get maximum index in each class
            max_j = maximum(class)

            #Update parameters of maximum index in the class
            δ_ntwk.neurons[max_j].β -= step*grads[max_j]
        end
        #Syncrhonize remaining parameters.
        synchronizeParameters!(δ_ntwk,partition)
    else
        update!(δ_ntwk, grads, step)
    end
    return δ_ntwk
end

"""
Returns full loss over entire training set (i.e. Σf(y,ŷ))
"""
function total_loss(ntwk :: Network.network, features, labels, loss :: Module)
    ΣF = 0.0

    for i=1:size(features,1)
        U, label = features[i], labels[i,:]
        # append initial hidden state (zero vector)
        U = vcat(U, zeros(ntwk.hid_dim))
        X = Network.evaluate(ntwk, U)[end-ntwk.results+1:end]
        ΣF += loss.evaluate(label, X, ntwk.results)
    end
    return ΣF
end

function total_loss(ntwk :: Network.network, δ_ntwk :: Network.network, features, labels, loss :: Module)
    ΣF = 0.0
    ΣδF = 0.0

    for i=1:size(features,1)
        U, label = features[i], labels[i,:]
        # append initial hidden state (zero vector)
        U = vcat(U, zeros(ntwk.hid_dim))
        X = Network.evaluate(ntwk, U)[end-ntwk.results+1:end]
        δX = Network.evaluate(δ_ntwk, U)[end-δ_ntwk.results+1:end]
        ΣF += loss.evaluate(label, X, ntwk.results)
        ΣδF += loss.evaluate(label, δX, δ_ntwk.results)
    end
    return ΣF, ΣδF
end

"""
Returns the deterministic gradient over the entire training set
 - Average gradient returned
"""
function total_gradient(ntwk :: Network.network, loss :: Module, penalty :: Module, features, labels, syncFlag :: Bool, partition, method :: String)

    batch_grad = []

    if syncFlag
        for i=1:size(features,1)
            U, label = vcat(features[i], zeros(ntwk.hid_dim)), labels[i,:]
            method == "adjoint" ?
				append!(batch_grad, [paramGrad(loss, ntwk, U, label, syncFlag, partition)]) : # adjoint update
				append!(batch_grad, [paramGrad(loss, penalty, ntwk, U, label, syncFlag, partition, ntwk_type)])
        end
    end

    return Dict(k => sum(map(i -> batch_grad[i][k], 1:length(batch_grad))) / length(batch_grad) for k in keys(batch_grad[1]))
end

"""
Computes optimal step size using backtracking line-search
	- α₀ = 1.0
	- τ = 0.5
	- m = ⟨∇F,∇F⟩ = ||∇F||²
	- p = ∇F
"""
function line_search!(ntwk :: Network.network, features, labels, loss :: Module, penalty :: Module, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, method :: String)

    ct = 0
    τ = 0.5
	α = 1.0
    update_grad = total_gradient(ntwk, loss, penalty, features, labels, syncFlag, partition, method)
    #update_grad = Helpers.batchGrad(grads)
    δ_ntwk = delta_network(ntwk, α, update_grad, syncFlag, partition)
    F, δF = total_loss(ntwk, δ_ntwk, features, labels, loss)

    # ||∇f(x)||^2
    m = sum(collect(Iterators.flatten(values(update_grad))).^2)
	t = 0.5*m
    while F - δF < α*t # until F-δF >= α*t
		# update ct and α
		ct += 1
        α = τ^ct*α
        δ_ntwk = delta_network(ntwk, α, update_grad, syncFlag, partition)
        δF = total_loss(δ_ntwk, features, labels, loss)
    end
    # update paramters
    update!(ntwk, update_grad, α, syncFlag, partition)

    return F, α
end


end #end module
