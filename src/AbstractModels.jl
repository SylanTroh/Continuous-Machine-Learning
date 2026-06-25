"""
The root model supertype. Every trainable model is an `AbstractModel`:

    AbstractModel
    ├── AbstractFlowModel       (stateless)
    │   └── NeuralFlowODE
    └── AbstractStatefulModel   (stateful)
"""
module AbstractModels

export AbstractModel, AbstractStatefulModel

abstract type AbstractModel end

# Stub supertype for stateful models (needs to implement its own integrator rather than using the shared solvers)
abstract type AbstractStatefulModel <: AbstractModel end

end
