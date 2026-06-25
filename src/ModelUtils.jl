"""
Model inspection and copying utilities. `param_vector`/`set_params!` convert between the
model and a flat ComponentVector with named blocks.
"""
module ModelUtils

using ComponentArrays
using ..AbstractRepresentation
using ..ODEModel
using ..BlockField: BlockSpec
using ..ResBlockModel: ResBlockFlow

export n_params, n_params_breakdown, copy_model
export param_vector, set_params!

"""
    n_params(model::NeuralFlowODE) → Int

Total number of parameters in the model.
"""
function n_params(model::NeuralFlowODE)
    nA = sum(length, vals(model.A))
    nB = sum(length, vals(model.B))
    return nA + nB + length(model.W_in) + length(model.W_out)
end

"""
    n_params_breakdown(model::NeuralFlowODE) → NamedTuple

Parameter counts per component.
"""
function n_params_breakdown(model::NeuralFlowODE)
    nA = sum(length, vals(model.A))
    nB = sum(length, vals(model.B))
    nWi = length(model.W_in)
    nWo = length(model.W_out)
    return (A=nA, B=nB, W_in=nWi, W_out=nWo, total=nA + nB + nWi + nWo)
end

"""
    copy_model(model::NeuralFlowODE) → NeuralFlowODE

Deep copy used for snapshots during training.
"""
function copy_model(model::NeuralFlowODE)
    A_new = reconstruct(model.A, [copy(Ak) for Ak in vals(model.A)])
    B_new = reconstruct(model.B, [copy(Bk) for Bk in vals(model.B)])
    return NeuralFlowODE(A_new, B_new, copy(model.W_in), copy(model.W_out), model.sigma)
end

"""
    param_vector(model::NeuralFlowODE) → ComponentVector

Copy all model parameters into one flat vector with named blocks:
`A` (d×d×nA), `B` (d×nB), `W_in`, `W_out`. Inverse of `set_params!`.
"""
function param_vector(model::NeuralFlowODE)
    return ComponentVector(
        A=cat(vals(model.A)...; dims=3),
        B=hcat(vals(model.B)...),
        W_in=copy(model.W_in),
        W_out=copy(model.W_out))
end

"""
    set_params!(model::NeuralFlowODE, p::ComponentVector) → model

Write the blocks of `p` back into the model. Inverse of `param_vector`.
"""
function set_params!(model::NeuralFlowODE, p::ComponentVector)
    for k in eachindex(vals(model.A))
        vals(model.A)[k] .= @view p.A[:, :, k]
    end
    for k in eachindex(vals(model.B))
        vals(model.B)[k] .= @view p.B[:, k]
    end
    model.W_in .= p.W_in
    model.W_out .= p.W_out
    return model
end

# ==== Residual block ====

"""
    n_params(model::ResBlockFlow) → Int
"""
n_params(model::ResBlockFlow) =
    model.spec.nA_total + model.spec.nB_total + length(model.W_in) + length(model.W_out)

"""
    n_params_breakdown(model::ResBlockFlow) → NamedTuple

A/B per-layer-summed coefficient counts plus the embeddings.
"""
function n_params_breakdown(model::ResBlockFlow)
    nA = model.spec.nA_total
    nB = model.spec.nB_total
    nWi = length(model.W_in)
    nWo = length(model.W_out)
    return (A=nA, B=nB, W_in=nWi, W_out=nWo, total=nA + nB + nWi + nWo)
end

"""
    copy_model(model::ResBlockFlow) → ResBlockFlow
"""
function copy_model(model::ResBlockFlow)
    As = [reconstruct(A, [copy(v) for v in vals(A)]) for A in model.As]
    Bs = [reconstruct(B, [copy(v) for v in vals(B)]) for B in model.Bs]
    return ResBlockFlow(model.spec, As, Bs, copy(model.W_in), copy(model.W_out), model.sigma)
end

"""
    param_vector(model::ResBlockFlow) → ComponentVector

Flat blocks `A`, `B`, `W_in`, `W_out`.
"""
function param_vector(model::ResBlockFlow)
    Aflat = reduce(vcat, (vec(cat(vals(A)...; dims=3)) for A in model.As))
    Bflat = reduce(vcat, (vec(reduce(hcat, vals(B))) for B in model.Bs))
    return ComponentVector(A=Aflat, B=Bflat, W_in=copy(model.W_in), W_out=copy(model.W_out))
end

"""
    set_params!(model::ResBlockFlow, p::ComponentVector) → model
"""
function set_params!(model::ResBlockFlow, p::ComponentVector)
    spec = model.spec
    ncp = spec.ncp
    for l in eachindex(model.As)
        m = spec.metas[l]
        Acoef = reshape(view(p.A, m.rangeA), m.wo, m.wi, ncp)
        for k in 1:ncp
            vals(model.As[l])[k] .= @view Acoef[:, :, k]
        end
        Bcoef = reshape(view(p.B, m.rangeB), m.wo, ncp)
        for k in 1:ncp
            vals(model.Bs[l])[k] .= @view Bcoef[:, k]
        end
    end
    model.W_in .= p.W_in
    model.W_out .= p.W_out
    return model
end

end
