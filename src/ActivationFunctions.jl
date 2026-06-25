"""
Activation functions σ and their derivatives.
"""
module ActivationFunctions

using ForwardDiff

export sigmoid, d_sigmoid, relu, d_relu, d_tanh
export activation_derivative

function sigmoid(z)
    return @. 1.0 / (1.0 + exp(-z))
end

function d_sigmoid(z)
    s = sigmoid(z)
    return @. s * (1.0 - s)
end

function relu(z)
    return @. max(0, z)
end

function d_relu(z)
    return @. ifelse(z > 0, one(eltype(z)), zero(eltype(z)))
end

# tanh(z) is in Base

function d_tanh(z)
    return @. Base.sech(z)^2
end

activation_derivative(::typeof(relu), z) = d_relu(z)
activation_derivative(::typeof(sigmoid), z) = d_sigmoid(z)
activation_derivative(::typeof(tanh), z) = d_tanh(z)
activation_derivative(σ, z) = ForwardDiff.derivative.(σ, z)

end
