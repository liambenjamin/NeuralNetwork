#Tools for verification and synchronization across parameters with (possible) equality partitions

module Partition

using LinearAlgebra, SparseArrays, Main.Neurons, Main.Network

import Main.Network: paramGrad, update!

export paramGrad, update!, adam_update!

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
function paramGrad(df :: Function, ntwk :: Network.network, feat :: Vector{Float64}, syncFlag :: Bool, partition :: Vector{Vector{Int64}} )
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
function update!(df :: Function,d2f :: Function, dg :: Function, ntwk :: Network.network, feat :: Vector{Float64}, step, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
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
function update!(loss :: Module, penalty :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label :: Vector{Float64}, step, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
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
function update!(loss :: Module, penalty :: Module, ntwk :: Network.network, feat :: Vector{Float64}, label, step, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, ntwk_type :: String)
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

function update!(ntwk :: Network.network, grads, step, syncFlag :: Bool, partition :: Vector{Vector{Int64}})
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
        update!(ntwk, grads, feat)
    end
end

function adam_update!(ntwk :: Network.network, grads, step, syncFlag :: Bool, partition :: Vector{Vector{Int64}}, m, v, iter)
    if syncFlag
        β1 = 0.9
        β2 = 0.99
        L = length(ntwk.neurons)
        ind = 1
        for class in partition

            #Get maximum index in each class
            max_j = maximum(class)

            # ADAM update #
            m[ind] = β1 * m[ind] + (1 - β1) * grads[max_j]
            v[ind] = β2 * v[ind] + (1 - β2) * grads[max_j].^2
            m_hat = m[ind] / (1 - β1^iter)
            v_hat = v[ind] / (1 - β2^iter)
            ntwk.neurons[max_j].β -= step * (m_hat ./ (sqrt.(v_hat) .+ 1e-10)) # ϵ=1e-10
            ind += 1
            ###########

            #Update parameters of maximum index in the class

            #ntwk.neurons[max_j].β -= step*grads[max_j]
        end

        #Syncrhonize remaining parameters.
        synchronizeParameters!(ntwk,partition)

    else
        update!(ntwk, grads, feat)
    end
end

end #end module
