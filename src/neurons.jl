#Author: Vivak Patel and Liam Johnston
#Date: February 22, 2018
#Description: An implementation of a variety of different (smooth) neurons (i.e., simple functions) and their derivatives.

module Neurons

using LinearAlgebra

export neuron, bias, linear, logistic, sigmoid, hypertan, hyptan, smoothmax, convolution, softmax, lstmCellState, lstmHiddenState, gruHiddenState, hadamardCellState

mutable struct neuron
  ide :: Int64    #Neuron identification number
  mod :: Symbol   #Symbol specifying module
  inp :: Int64    #Input dimension
  par :: Int64    #Parameter dimension
  out :: Int64    #Output dimension
  met :: Any      #Meta Information
  β :: Vector     #Parameter Vector
end

module bias
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    inp != 1 && @warn "Only first value of input will be used."
    par != 1 && @error "Parameter must be one dimensional."
    out_val = out()
    return neuron(ide, :bias, inp, par, out_val, met, 0.001*randn(1))
  end

  """Activation function"""
  act(X,β; kwargs...) = X[1]+β[1]

  """Output dimension"""
  out(args...; kwargs...) = 1

  """
  The gradient of the bias activation function with respect to the parameter.
  """
  dP(X,β; kwargs...) = ones(1)

  """
  The gradient of the bias activation function with respect to the input.
  """
  dX(X,β; kwargs...) = ones(1)

  """
  The second derivative of the bias activation function with respect to the input.
  """
  dXX(X,β; kwargs...) = zeros(1,1)

  """
  The second derivative of the bias activation function with respect to the parameter.
  """
  dPP(X,β; kwargs...) = zeros(1,1)

  """
  The second-order mixed derivative of the bias activation function. (DX then Dβ, not that it matters here.)
  """
  dXP(X,β; kwargs...) = zeros(1,1)

end #End Module


"""
Implements a neuron with a linear activation function.
"""
module linear
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Input must be one dimension smaller than parameter"
    out_val = out()
    return neuron(ide, :linear, inp, par, out_val, met, 0.001*randn(par))
  end

  """
  The linear activation function. Given a parameter, β, and an input, X, the output is β[1] + dot(β[2:end],X).
  """
  act(X,β; kwargs...) = β[1] + dot(β[2:end],X)

  """
  Returns the output dimension of the linear activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the linear activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    G = zeros(length(β))
    G[1] = 1
    G[2:end] = X
    return G
  end

  """
  The gradient of the linear activation function with respect to the input.
  """
  dX(X,β; kwargs...) = β[2:end]

  """
  The second derivative of the linear activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    dim_X = length(X)
    return zeros(dim_X,dim_X)
  end

  """
  The second derivative of the linear activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    P = length(β)
    return zeros(P,P)
  end

  """
  The second-order mixed derivative of the linear activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    dim_X = length(X)
    P = length(β)
    return hcat(zeros(dim_X),I)
  end

end #End Module

"""
Implements a logistic activation function
"""
module logistic
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp + 2 && @error "Parameter must be two dimensions larger than input"
    out_val = out()
    return neuron(ide, :logistic, inp, par, out_val, met, 0.001*randn(par))
  end

  """
  The logistic activation function with scaling. Given an input X and a parameter β, returns β[1]/(1 + exp(-β[2]- dot(X,β[3:end]))).
  """
  function act(X,β; kwargs...)
    return β[1]/(1 + exp(-β[2]- dot(X,β[3:end])))
  end

  """
  Returns the output dimension of the logistic activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the logistic activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    P = length(β)
    G = zeros(P)
    η = β[2] + dot(X,β[3:end])

    #W.R.T β[1]
    G[1] = 1/(1+exp(-η))
    G[2] = β[1]*G[1]*(1-G[1])
    G[3:end] = G[2]*X

    return G
  end

  """
  The gradient of the logistic activation function with respect to the input.
  """
  function dX(X,β; kwargs...)
    f = 1/(1 + exp(-β[2]-dot(β[3:end],X)))
    return (β[1]*f*(1-f))*β[3:end]
  end

  """
  Implements the second derivative of the logistic activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    f = 1/(1 + exp(-β[2]-dot(β[3:end],X)))
    ∂f_∂η = f*(1-f)
    return (β[1]*(1-2*f)*∂f_∂η)*(β[3:end]*β[3:end]')
  end

  """
  Implements the second derivative of the logistic activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    f = 1/(1 + exp(-β[2]-dot(β[3:end],X)))
    ∂f_∂η = f*(1-f)
    J1 = vcat(0, f*(1-f), f*(1-f)*X)
    J2 = vcat(∂f_∂η, β[1]*(1-2*f)*∂f_∂η, (β[1]*(1-2*f)*∂f_∂η)*X)*vcat(1,X)'
    return hcat(J1,J2)
  end

  """
  Implements the second-order mixed derivative of the logistic activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    # 'I' automatically deduces dimensions for identity matrix
    f = 1/(1 + exp(-β[2]-dot(β[3:end],X)))
    ∂f_∂η = f*(1-f)
    ∂G_∂η = -β[1]*(1-2*f)*∂f_∂η*β[3:end]
    J1 = f*(1-f)*β[3:end]
    J2 = -∂G_∂η
    J3 = β[1]*f*(1-f)*I - ∂G_∂η*X'
    return hcat(J1,J2,J3)
  end

end #End Module


"""
Implements a sigmoid activation function on the linear combination, Z
1. Z = W_{rec} × x_{k-1} + W_{in} × u_{k} + b
2. W_{rec} and W_{in} constitute a single row of the previously defined recurrent and input matrices
Options:
  - inp: length(x_{k-1}) + length(u_{k})
  - par: inp + 1 (bias term)
"""
module sigmoid
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Parameter must be one dimension larger than input."
    out_val = out()
    return neuron(ide, :sigmoid, inp, par, out_val, met, 1.0*randn(par))
  end

  """
  The sigmoid activation function. Given an input X and a parameter β, ( 1 / (1 + exp(βX)) ).
  """
  function act(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    return 1.0 ./ (1.0 .+ exp.(-Z))
  end

  """
  Returns the output dimension of the sigmoid activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the sigmoid activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    ∂Z = exp(-Z) / (1+exp(-Z))^2
    ∂P = vcat(1,X)
    return ∂P * ∂Z
  end

  """
  Implements the gradient of the sigmoid activation function with respect to the input.
  """
  function dX(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    ∂Z = exp(-Z) / (1+exp(-Z))^2
    ∂X = β[2:end]
    return ∂X * ∂Z
  end

  """
  Implements the second derivative of the sigmoid activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    J = (2*exp(-2*Z)) / (1+exp(-Z))^3 - exp(-Z) / (1+exp(-Z))^2
    return β[2:end] * J * β[2:end]'
  end

  """
  Implements the second derivative of the sigmoid activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    J = (2*exp(-2*Z)) / (1+exp(-Z))^3 - exp(-Z) / (1+exp(-Z))^2
    return vcat(1,X) * J * vcat(1,X)'
  end

  """
  Implements the second-order mixed derivative of the sigmoid activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    # ∂X
    Z = β[1] + dot(β[2:end],X)
    T1 = diagm(0 => ones(length(β))) * exp(-Z) * (1+exp(-Z))^-2
    T2 = -β * exp(-Z) * (1+exp(-Z))^-2 * vcat(1,X)'
    T3 = β * exp(-Z) * 2/(1+exp(-Z))^3 * exp(-Z) * vcat(1,X)'
    ∂XP = (T1 + T2 + T3)[2:end,:]
    return ∂XP
  end

end #End Module

"""
Implements a hyperbolic tangent activation function on the linear combination, Z
1. Z = W_{rec} × x_{k-1} + W_{in} × u_{k} + b
2. W_{rec} and W_{in} constitute a single row of the previously defined recurrent and input matrices
Options:
  - inp: length(x_{k-1}) + length(u_{k})
  - par: inp + 1 (bias term)
"""
module hypertan
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Parameter must be one dimension larger than input."
    out_val = out()
    return neuron(ide, :hypertan, inp, par, out_val, met, 1.0*randn(par))
  end

  """
  The tanh activation function. Given an input X and a parameter β, ( 2 / (1 + exp(-2βX)) - 1 ).
  """
  function act(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    return 2.0 / (1.0 + exp(-2*Z)) - 1.0
  end

  """
  Returns the output dimension of the tanh activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the tanh activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    ∂P = 4 * (1+exp(-2*Z))^-2 * vcat(1.0,X) * exp(-2*Z)
    return ∂P
  end

  """
  Implements the gradient of the tanh activation function with respect to the input.
  """
  function dX(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    ∂X = 4 * (1+exp(-2*Z))^-2 * β[2:end] * exp(-2*Z)
    return ∂X
  end

  """
  Implements the second derivative of the tanh activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    J1 = 16 * β[2:end] * (1.0 + exp(-2*Z))^-3 * exp(-4*Z) * β[2:end]'
    J2 = -8 * β[2:end] * (1.0 + exp(-2*Z))^-2 * exp(-2*Z) * β[2:end]'
    return J1 + J2
  end

  """
  Implements the second derivative of the tanh activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    J1 = 16 * vcat(1,X) * (1.0 + exp(-2*Z))^-3 * exp(-2*Z)^2 * vcat(1,X)'
    J2 = -8 * vcat(1,X) * (1.0 + exp(-2*Z))^-2 * exp(-2*Z) * vcat(1,X)'
    return J1 + J2
  end

  """
  Implements the second-order mixed derivative of the tanh activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    Z = β[1] + dot(β[2:end],X)
    J1 = 16 * β[1:end] * (1.0 + exp(-2*Z))^-3 * exp(-4*Z) * vcat(1,X)'
    J2 = 4 * (1.0 + exp(-2*Z))^-2 * exp(-2*Z) * diagm(0 => ones(length(β[1:end])))
    J3 = -8 * β[1:end] * (1.0 + exp(-2*Z))^-2 * exp(-2*Z) * vcat(1,X)'
    ∂XP = J1 + J2 + J3
    return ∂XP[2:end,:]
  end

end #End Module

"""
Implements a neuron with a hyperbolic tangent activation function
"""
module hyptan
  using LinearAlgebra
  import ..Neurons: neuron

  """
  Given an input, initializes a parameter of the appropriate dimension for the hyperbolic tangent activation function.
  """
  function init(ide,inp,par; met = nothing)
    par != inp+3 && @error "Input must be three dimensions smaller than parameter"
    out_val = out()
    return neuron(ide, :hyptan, inp, par, out_val, met, 0.001*randn(par))
  end

  """
  The hyperbolic tangent activation function. Given a parameter, β, and an input, X, the output is β[1]+β[2]tanh(β[3] + dot(β[4:end],X))
  """
  function act(X,β; kwargs...)
    return β[1]+β[2]*tanh(β[3] + dot(β[4:end],X))
  end

  """
  Returns the output dimension of the hyperbolic tangent activation function.
  """
  out(args...;kwargs...) = 1

  """
  The gradient of the hyperbolic tangent activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    G = zeros(length(β))
    η = β[3] + dot(β[4:end],X)
    G[1] = 1
    G[2] = tanh(η)
    G[3] = β[2]*sech(η)^2
    G[4:end] = G[3]*X
    return G
  end

  """
  The gradient of the hyperbolic tangent activation function with respect to the input.
  """
  function dX(X,β; kwargs...)
    η = β[3] + dot(β[4:end],X)
    G = (β[2]*sech(η)^2)*β[4:end]
    return G
  end

  """
  The second derviative of the hyperbolic tangent activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    η = β[3] + dot(β[4:end],X)
    return (-2*β[2]*tanh(η)*sech(η)^2)*(β[4:end]*β[4:end]')
  end

  """
  The second derivative of the hyperbolic tangent activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    I = length(X)
    P = length(β)
    η = β[3] + dot(β[4:end],X)
    g = -2*β[2]*tanh(η)*sech(η)^2
    J1 = zeros(P)
    J2 = vcat(zeros(2), sech(η)^2, sech(η)^2*X)
    J3 = vcat(zeros(1,I+1),
          sech(η)^2*hcat(1.0,X'),
          g*hcat(1.0, X'),
          g*hcat(X, X*X'))
    return hcat(J1,J2,J3)
  end

  """
  Implements the second-order mixed derivative of the hyperbolic tangent activation function with respect to the parameter. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    # 'I' automatically deduced dimensions for identity matrix
    η = β[3] + dot(β[4:end],X)
    ∂G_∂η = (-2*β[2]*tanh(η)*sech(η)^2)*β[4:end]
    J1 = zeros(length(X))
    J2 = sech(η)^2*β[4:end]
    J3 = ∂G_∂η
    J4 = β[2]*sech(η)^2*I + ∂G_∂η*X'
    return hcat(J1,J2,J3,J4)
  end

end #End Module


"""
Implements a smoothmax pooling activation function
"""
module smoothmax
  using LinearAlgebra
  import ..Neurons: neuron

  """
  Initializes the smoothmax activation function. (If the parameter is negative, it is the softmin activation function).
  """
  function init(ide, inp, par; met = nothing)
    par != 1 && @error "Parameter must be one dimensional"
    out_val = out()
    return neuron(ide, :smoothmax, inp, par, out_val, met, 0.001*randn(par))
  end

  """
  The smoothmax activation function. Given an input X and a parameter β, returns sum( X .* exp.(β*X))/sum( exp.(β*X)).
  """
  function act(X,β; kwargs...)
      Z = exp.(β[1]*X)
      return sum(X.*Z)/sum(Z)
  end

  """
  Returns the output dimension of the smooth max activation function.
  """
  out(args...;kwargs...) = 1

  """
  The gradient of the smoothmax activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
      dim_X = length(X)
      Z = exp.(β[1]*X)
      S = sum(Z)
      ∂f_∂Z = (X/S - dot(X,Z)/S^2*ones(dim_X))
      return Float64[dot(∂f_∂Z, X.*Z)]
  end

  """
  The gradient of the smoothmax activation function with respect to the input.
  """
  function dX(X,β; kwargs...)
      Z = exp.(β[1]*X)
      S = sum(Z)
      ∂Z_∂X = β[1]*Z
      return 1/S*(Z + ∂Z_∂X.*X) - dot(X,Z)/S^2*∂Z_∂X
  end

  """
  The second derivative of the smoothmax activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
      Z = exp.(β[1]*X)
      S = sum(Z)
      ∂Z = β[1]*Z #w.r.t X
      ∂Z2 = β[1]*∂Z #w.r.t X
      vc = 1/S*(2*∂Z + ∂Z2.*X) - dot(X,Z)/S^2*∂Z2
      outer = (-1/S^2)*((Z + ∂Z.*X)*(∂Z)' + ∂Z*(Z + ∂Z.*X)')  + (2*dot(X,Z)/S^3)*(∂Z*∂Z')
      #return diagm(vc) + outer
      return diagm(0=>vc) + outer
  end

  """
  The second derivative of the smoothmax activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
      Z = exp.(β[1]*X)
      S = sum(Z)
      ∂Z = X.*Z #w.r.t β
      S2 = sum(∂Z)
      ∂Z2 = (X.^2).*Z
      S3 = sum(∂Z2)
      out = zeros(1,1)
      out[1,1] = dot(X, ∂Z2)/S - 2*dot(X,∂Z)*S2/S^2 - dot(X,Z)/S^2*(S3 - 2/S*S2^2)
      return out
  end

  """
  The second-order mixed derivative of the smoothmax activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
      dim_X = length(X)
      Z = exp.(β[1]*X)
      S = sum(Z)
      ∂Z_∂X = β[1]*Z
      ∂Z_∂β = X.*Z
      Sβ = sum(∂Z_∂β)
      ∂²Z_∂X∂β = Z + β[1]*∂Z_∂β
      t1 = 1/S*(∂Z_∂β + ∂²Z_∂X∂β.*X)
      t2 = -Sβ/S^2*(Z + ∂Z_∂X.*X)
      t3 = (-dot(X,Z)/S^2)*∂²Z_∂X∂β
      t4 = (-dot(X,∂Z_∂β)/S^2)*∂Z_∂X
      t5 = (2*dot(X,Z)*Sβ/S^3)*∂Z_∂X
      out = zeros(dim_X,1)
      out[:,1] = t1+t2+t3+t4+t5
      return out
  end
end #End Module

"""
Implements a neuron with a one-dimensional convolution activation function (no bias)
"""
module convolution
  using LinearAlgebra
  using SparseArrays
  import ..Neurons: neuron

  """
  Initialization function

  `met` keyword argument is a vector.

  - First element specifies stride length of the one-dimensional convolution
  - Second element specifies gap between each parameter
  - Third argument specifies maximum number of steps to take
  """
  function init(ide, inp, par; met :: Vector)
    out_val = out(inp, par, met = met)
    return neuron(ide, :convolution, inp, par, out_val, met, 0.001*randn(par))
  end

  """
  Reshapes input vector `X` into an appropriately dimensioned matrix that generates output.
  """
  function transform(X, β, met)
    strd, skp, stps = met
    par = length(β)
    out_dim = out(length(X),par; met=met)
    Z = hcat(map(λ -> X[(1 + (λ-1)*(skp+1)):strd:(1 + (λ-1)*(skp+1) + (out_dim-1)*strd)] ,1:par)...)
  end

  """Activation function"""
  act(X,β; met) = transform(X,β,met)*β

  """Output dimension"""
  function out(inp, par; met)
    strd, skp, stps = met
    if par == 1
      #One dimensional case; 1 + strd*N <= inp, solve for integer N
      return convert(Int64,min(div(inp-1,strd), stps))
    else
      #Multi-dimensional case; 1 + (par-1)*(skp+1) + strd*N <= inp, solve for N
      return convert(Int64,min(div(inp-1 - (par-1)*(skp+1), strd), stps))
    end
  end

  """Jacobian with respect to parameter"""
  dP(X,β; met) = transform(X,β,met)

  """Jacobian with respect to input"""
  function dX(X,β; met)
    strd, skp, stps = met
    par = length(β)
    inp = length(X)
    out_dim = out(inp, par; met=met)
    gra = spzeros(out_dim,inp)
    for i = 1:out_dim
      indx = (1 + (i-1)*strd):(skp+1):(1 + (i-1)*strd + (par-1)*(skp+1))
      gra[i,indx] = β
    end
    return gra
  end

  """Second derivative with respect to input"""
  function dXX(X,β; met)
    inp = length(X)
    par = length(β)
    out_dim = out(inp, par, met=met)
    return repeat([spzeros(inp, inp)], out_dim) #Repeats copies of pointer for same matrix!
  end

  dXX(X,β, ind; met) = spzeros(length(X),length(X))

  """Second derivative with respect to parameter"""
  function dPP(X,β; met)
    inp = length(X)
    par = length(β)
    out_dim = out(inp, par, met=met)
    return repeat([spzeros(par,par)], out_dim) #Repeats copies of pointer for same matrix!
  end

  dPP(X,β, ind; met) = spzeros(length(β),length(β))

  """Second-order mixed derivative"""
  function dXP(X,β; met)
    strd, skp, stps = met
    par = length(β)
    inp = length(X)
    out_dim = out(inp, par; met=met)
    hes = [spzeros(inp,par) for i = 1:out_dim]
    for i = 1:out_dim
      indx = (1 + (i-1)*strd):(skp+1):(1 + (i-1)*strd + (par-1)*(skp+1))
      for j = 1:par
        hes[i][indx[j],j] = 1
      end
    end
    return hes
  end
  function dXP(X,β, ind; met)
    strd, skp, stps = met
    par = length(β)
    inp = length(X)
    out_dim = out(inp, par; met=met)
    hes = spzeros(inp,par)
    indx = (1 + (ind-1)*strd):(skp+1):(1 + (ind-1)*strd + (par-1)*(skp+1))
    for j = 1:par
      hes[indx[j],j] = 1
    end
    return hes
  end

end #End Module

"""
Implements softmax activation and derivatives
"""
module softmax
  using LinearAlgebra, Statistics
  import ..Neurons: neuron

  """
  Initializes the softmax activation function.
  """
  function init(ide, inp, par; met = nothing)
    par != 1 && @error "Parameter must be one dimensional"
    out_val = out(inp)
    return neuron(ide, :softmax, inp, par, out_val, met, 0.0*randn(par))
  end

  """
  The softmax activation function. Given an input X, returns exp.(X) /sum( exp.(X)).
  """
  function act(X, β; kwargs...)
    """
    Note: maximum(X) subtracted from each element of X to induce numerical stability
    """
    Z = X .- maximum(X)
    return exp.(Z) / sum(exp.(Z))
  end

  """
  The gradient of the softmax activation function with respect to the input.
  """
  function dX(X,β; kwargs...)
    L = length(X)
    S = sum(exp.(X))
    ∂X = reshape(collect(Iterators.flatten(map(i -> -exp.(X[i] .+ X) / S^2, 1:L))), L, L)
    ∂X[diagind(∂X)] += exp.(X) / S
    return ∂X
  end

  """
  Returns the output dimension of the smooth max activation function.
  """
  out(inp;kwargs...) = inp

  """
  The second derivative of the softmax activation function with respect to the input.
  """

  function dXX(X, β; kwargs...)
    L = length(X)
    S = sum(exp.(X))
    ∂XX = zeros(L,L,L)
    for (i,j,k) in Iterators.product(1:L,1:L,1:L)
      ∂XX[i,j,k] = i == j ?
        (j == k ?
          dX(X,β)[j,k] - (2*exp(2*X[k]) / S^2) + (2*exp(3*X[k]) / S^3) :
          dX(X,β)[j,k] + (2*exp(2*X[j]+X[k]) / S^3)) :
        (i == k || j == k ?
          (-exp(X[i]+X[j]) / S^2) + (2*exp(X[i]+X[j]+X[k]) / S^3) :
          2*exp(X[i]+X[j]+X[k]) / S^3)
    end

    return ∂XX
  end

  """
  Jacobian of softmax neuron wrt parameter (--> 0)
  """
  function dP(X,β; kwargs...)
    return zeros(length(X))
  end

  """
  Second-order derivative with respect to the parameter (--> 0 (1×1))
  """
  function dPP(X,β; kwargs...)
    return zeros(length(X),length(X))
  end

  """
  Second-order mixed derivative
  """
  function dXP(X,β; kwargs...)
    return zeros(length(X), length(β))
  end

end #End Module

"""
Implements the LSTM cell state activation function on the linear combination
  Input: cκ = fκ × cκ₋₁ + iκ × c̃κ
"""
module lstmCellState
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Parameter must be one dimension larger than input."
    out_val = out()
    return neuron(ide, :lstmCellState, inp, par, out_val, met, 1.0*randn(par))
  end

  """
  The lstm cell state activation function. Given an input X and a parameter β
    Note: X[1] = fκ
          X[2] = cκ₋₁
          X[3] = iκ
          X[4] = c̃κ
  """
  function act(X,β; kwargs...)
    c = X[1]*X[2] + X[3]*X[4]
    return c
  end

  """
  Returns the output dimension of the sigmoid activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the sigmoid activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    return zeros(length(β))
  end

  """
  Implements the gradient of the lstm cell state activation function with respect to the input.
    Returns: ∂X = [∂f, ∂cκ₋₁, ∂i, ∂c̃κ]
  """
  function dX(X,β; kwargs...)
      ∂f = X[2]
      ∂c = X[1]
      ∂i = X[4]
      ∂c̃ = X[3]
    return [∂f, ∂c, ∂i, ∂c̃]
  end

  """
  Implements the second derivative of the lstm cell state activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    J = zeros(length(X),length(X))
    J[1,2] = 1.0
    J[2,1] = 1.0
    J[3,4] = 1.0
    J[4,3] = 1.0
    return J
  end

  """
  Implements the second derivative of the lstm cell state activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    return zeros(length(β), length(β))
  end

  """
  Implements the second-order mixed derivative of the lstm cell state activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    return zeros(length(X), length(β))
  end

end #End Module


"""
Implements the LSTM hidden state activation function on the linear combination
  Input: xκ = oκ × σ(cκ)
"""
module lstmHiddenState
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Parameter must be one dimension larger than input."
    out_val = out()
    return neuron(ide, :lstmHiddenState, inp, par, out_val, met, 1.0*randn(par))
  end

  """
  The lstm hidden state activation function.
    xκ = oκ × σ(cκ) where σ is the hyperbolic tangent function
    Note: X[1] = oκ
          X[2] = cκ
  """
  function act(X,β; kwargs...)
    o = X[1]
    σ_c = (2.0 / (1+exp(-2*X[2])) ) - 1.0
    return o * σ_c
  end

  """
  Returns the output dimension of the lstm hidden state activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the lstm hidden state activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    return zeros(length(β))
  end

  """
  Implements the gradient of the lstm hidden state activation function with respect to the input.
    Returns: ∂X = [∂o, ∂c]
  """
  function dX(X,β; kwargs...)
    ∂o = (2.0 / (1+exp(-2*X[2])) ) - 1.0
    ∂c = (4*X[1]*exp(-2*X[2]) ) / (1.0+exp(-2*X[2]))^2
    return [∂o, ∂c]
  end

  """
  Implements the second derivative of the lstm hidden state activation function with respect to the input.

  """
  function dXX(X,β; kwargs...)
    J = zeros(2,2)
    D = 1+exp(-2*X[2])
    J[1,1] = 0.0
    J[1,2] = ( 4.0*exp(-2*X[2]) ) / D^2
    J[2,1] = ( 4.0*exp(-2*X[2]) ) / D^2
    J[2,2] = ( (-8*X[1]*exp(-2*X[2])) / D^2 ) + ( (16*exp(-2*X[2])*X[1]*exp(-2*X[2])) / D^3 )
    return J
  end

  """
  Implements the second derivative of the lstm hidden state activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    return zeros(length(β), length(β))
  end

  """
  Implements the second-order mixed derivative of the lstm hidden state activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    return zeros(length(X), length(β))
  end

end #End Module

"""
Implements the hadamard cell state activation function on the linear combination
- Activation: hκ = rκ ∘ xκ₋₁
- Input: rκ and xκ₋₁ (both as 1-dimension)
"""
module hadamardCellState
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Parameter must be one dimension larger than input."
    out_val = out()
    return neuron(ide, :hadamardCellState, inp, par, out_val, met, 1.0*randn(par))
  end

  """
  The hadamard cell state activation function.
    hκ = rκ ∘ xκ₋₁ where ∘ is the hadamard product
    Note: rκ = X[1]
          xκ₋₁ = X[2]
  """
  function act(X,β; kwargs...)
    r, x_prev = X[1], X[2]
    return r * x_prev
  end

  """
  Returns the output dimension of the hadamard cell state activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the hadamard cell state activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    return zeros(length(β))
  end

  """
  Implements the gradient of the hadamard hidden state activation function with respect to the input.
    Returns: ∂X = [∂rκ, ∂xκ₋₁]
  """
  function dX(X,β; kwargs...)
    r, x_prev = X[1], X[2]
    ∂r = x_prev
    ∂x_prev = r
    return [∂r, ∂x_prev]
  end

  """
  Implements the second derivative of the hadamard hidden state activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    J = zeros(length(X),length(X))
    J[1,2] = 1.0
    J[2,1] = 1.0
    return J
  end

  """
  Implements the second derivative of the hadamard cell state activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    return zeros(length(β), length(β))
  end

  """
  Implements the second-order mixed derivative of the hadamard cell state activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    return zeros(length(X), length(β))
  end

end #End Module

"""
Implements the GRU hidden state activation function on the linear combination
- Activation: xκ = (1-zκ)∘xκ₋₁ + zκ∘hκ
"""
module gruHiddenState
  using LinearAlgebra
  import ..Neurons: neuron

  function init(ide, inp, par; met = nothing)
    par != inp+1 && @error "Parameter must be one dimension larger than input."
    out_val = out()
    return neuron(ide, :gruHiddenState, inp, par, out_val, met, 1.0*randn(par))
  end

  """
  The gru hidden state activation function.
    xκ = (1-zκ)∘xκ₋₁ + zκ∘hκ where ∘ is the hadamard product and variables are 1-dimensional
    Note: X[1] = zκ
          X[2] = xκ₋₁
          X[3] = hκ
  """
  function act(X,β; kwargs...)
    z, x_prev, h = X[1], X[2], X[3]
    return (1-z)*x_prev + z*h
  end

  """
  Returns the output dimension of the gruHiddenState activation function.
  """
  out(args...; kwargs...) = 1

  """
  The gradient of the gruHiddenState activation function with respect to the parameter.
  """
  function dP(X,β; kwargs...)
    return zeros(length(β))
  end

  """
  Implements the gradient of the gru hidden state activation function with respect to the input.
    Returns: ∂X = [∂z, ∂xκ₋₁, ∂h]
  """
  function dX(X,β; kwargs...)
    z, x_prev, h = X[1], X[2], X[3]
    ∂z = -x_prev + h
    ∂x_prev = 1-z
    ∂h = z
    return [∂z, ∂x_prev, ∂h]
  end

  """
  Implements the second derivative of the gru hidden state activation function with respect to the input.
  """
  function dXX(X,β; kwargs...)
    z, x_prev, h = X[1], X[2], X[3]
    J = zeros(length(X),length(X))
    J[1,2] = -1.0
    J[1,3] = 1.0
    J[2,1] = -1.0
    J[3,1] = 1.0
    return J
  end

  """
  Implements the second derivative of the gru hidden state activation function with respect to the parameter.
  """
  function dPP(X,β; kwargs...)
    return zeros(length(β), length(β))
  end

  """
  Implements the second-order mixed derivative of the gru hidden state activation function. (DX then Dβ)
  """
  function dXP(X,β; kwargs...)
    return zeros(length(X), length(β))
  end

end #End Module


"""
Numerical differentiation function
"""
function differenceDerivative(X, β, keyArgs :: Dict, func :: Function; ϵ = 1e-8)
  base = func(X, β; keyArgs...)
  inp = length(X)
  par = length(β)
  if typeof(base) <: Number #Scalar output from func
    gX = zeros(inp)
    for j in 1:inp
      y = X[j]
      inp == 1 ? X += ϵ : X[j] += ϵ
      gX[j] = (func(X,β; keyArgs...) - base)/ϵ
      inp == 1 ? X = y : X[j] = y
    end

    gP = zeros(par)
    for j in 1:par
      y = β[j]
      par == 1 ? β += ϵ : β[j] += ϵ
      gP[j] = (func(X,β; keyArgs...) - base)/ϵ
      par == 1 ? β = y : β[j] = y
    end

    return Dict(:X => gX, :P => gP)
  elseif typeof(base) <: Vector #Vector Output from func
    jX = zeros(length(base), inp)
    for j in 1:inp
      y = X[j]
      inp == 1 ? X += ϵ : X[j] += ϵ
      jX[:,j] = (func(X,β; keyArgs...) - base)/ϵ
      inp == 1 ? X = y : X[j] = y
    end

    jP = zeros(length(base), par)
    for j in 1:par
      y = β[j]
      par == 1 ? β += ϵ : β[j] += ϵ
      jP[:,j] = (func(X,β; keyArgs...) - base)/ϵ
      par == 1 ? β = y : β[j] = y
    end

    return Dict(:X => jX, :P => jP)
  else #Matrix Output from func
    out_dim = size(base)[1]
    m = size(base)[2]
    tX = zeros( out_dim, m, inp)
    for j in 1:inp
      y = X[j]
      inp == 1 ? X += ϵ : X[j] += ϵ
      tX[:,:,j] = (func(X,β; keyArgs...) - base)/ϵ
      inp == 1 ? X = y : X[j] = y
    end

    tP = zeros( out_dim, m, par)
    for j in 1:par
      y = β[j]
      par == 1 ? β += ϵ : β[j] += ϵ
      tP[:,:,j] = (func(X,β; keyArgs...) - base)/ϵ
      par == 1 ? β = y : β[j] = y
    end

    return Dict(:X => tX, :P => tP)
  end
end

"""
Numerical differentiation function for bias inclusion in β
"""
#num_der = Neurons.differenceDerivativeBias(params[model][1],params[model][2], params[model][3], eval(model).act)
function differenceDerivativeBias(X, β, keyArgs :: Dict, func :: Function; ϵ = 1e-8)
  base = func(X, β; keyArgs...)
  #base = func(X, β; keyArgs)
  inp = length(X)
  par = length(β)
  if typeof(base) <: Number #Scalar output from func
    gX = zeros(inp)
    for j in 1:inp
      y = X[j]
      inp == 1 ? X += ϵ : X[j] += ϵ
      gX[j] = (func(X,β; keyArgs...) - base)/ϵ
      inp == 1 ? X = y : X[j] = y
    end

    gP = zeros(par)
    for j in 1:par
      y = β[j]
      par == 1 ? β += ϵ : β[j] += ϵ
      gP[j] = (func(X,β; keyArgs...) - base)/ϵ
      par == 1 ? β = y : β[j] = y
    end

    return Dict(:X => gX, :P => gP)
  elseif typeof(base) <: Vector #Vector Output from func
    jX = zeros(length(base), inp)
    for j in 1:inp
      y = X[j]
      inp == 1 ? X += ϵ : X[j] += ϵ
      jX[:,j] = (func(X,β; keyArgs...) - base)/ϵ
      inp == 1 ? X = y : X[j] = y
    end

    jP = zeros(length(base), par)
    for j in 1:par
      y = β[j]
      par == 1 ? β += ϵ : β[j] += ϵ
      jP[:,j] = (func(X,β; keyArgs...) - base)/ϵ
      par == 1 ? β = y : β[j] = y
    end

    return Dict(:X => jX, :P => jP)
  else #Matrix Output from func
    out_dim = size(base)[1]
    m = size(base)[2]
    tX = zeros( out_dim, m, inp)
    for j in 1:inp
      y = X[j]
      inp == 1 ? X += ϵ : X[j] += ϵ
      tX[:,:,j] = (func(X,β; keyArgs...) - base)/ϵ
      inp == 1 ? X = y : X[j] = y
    end

    tP = zeros( out_dim, m, par)
    for j in 1:par
      y = β[j]
      par == 1 ? β += ϵ : β[j] += ϵ
      tP[:,:,j] = (func(X,β; keyArgs...) - base)/ϵ
      par == 1 ? β = y : β[j] = y
    end

    return Dict(:X => tX, :P => tP)
  end
end

end #End Module Neurons
