"""
    BlockField

Allow stringing multiple nonlinear layers together into one block.

    F(x,t) = σ(A_N(t)·σ(… σ(A_1(t)·x + B_1(t)) …) + B_N(t)).

A block is `BlockSpec(d, ncp, mults)`, where `mults` are the width multipliers of dimension `d`.
Each block maps `d→d`, so the last entry must always be `1//1`.

- `[1//1]` one activation per block
- `[1//1, 1//1]` two activations per block
- `[1//4, 1//4, 1//1]` ResNet bottleneck block (d → d/4 → d/4 → d)
"""
module BlockField

using LinearAlgebra
using ..ActivationFunctions
using ..AbstractRepresentation: device_basis_weights

export BlockSpec, build_block, cpu_field_ops, layer_widths, block_nparams

"""
    LayerMeta

Per-layer shape and flat index offsets.
"""
struct LayerMeta
    idx::Int
    wo::Int
    wi::Int
    rangeA::UnitRange{Int}
    rangeB::UnitRange{Int}
end

"""
    BlockSpec(d, ncp, mults)
"""
struct BlockSpec
    d::Int
    ncp::Int
    mults::Vector{Rational{Int}}
    metas::Vector{LayerMeta}
    nA_total::Int
    nB_total::Int
end

function BlockSpec(d::Integer, ncp::Integer, mults::AbstractVector{<:Rational})
    isempty(mults) && error("block must have at least one layer")
    last(mults) == 1 || error("final layer multiplier must be 1//1 (output width = d for the residual add); got $(last(mults))")
    for (l, m) in enumerate(mults)
        isinteger(m * d) || error("layer $l width $m·$d = $(m*d) is not an integer; choose d divisible by denominator($m) = $(denominator(m))")
    end
    wos = Int.(mults .* d)
    wis = vcat(d, wos[1:end-1])               # layer l input = previous output; layer 1 input = d
    metas = LayerMeta[]
    a = 0
    b = 0
    for l in eachindex(mults)
        la = wos[l] * wis[l] * ncp
        lb = wos[l] * ncp
        push!(metas, LayerMeta(l, wos[l], wis[l], (a+1):(a+la), (b+1):(b+lb)))
        a += la
        b += lb
    end
    BlockSpec(Int(d), Int(ncp), collect(Rational{Int}, mults), metas, a, b)
end

"""
    layer_widths(spec) → Vector{Tuple{Int,Int}}

Per-layer `(w_out, w_in)`.
"""
layer_widths(spec::BlockSpec) = [(m.wo, m.wi) for m in spec.metas]

"""
    block_nparams(spec) → (A, B) total flat coefficient counts.
"""
block_nparams(spec::BlockSpec) = (A=spec.nA_total, B=spec.nB_total)

"""
    build_block(spec, σ, ops) → block_eval

Compile the block spec into a specific VJP
"""
function build_block(spec::BlockSpec, σ, ops)
    metas = spec.metas
    N = length(metas)
    return function block_eval(X, p, t)
        wA = ops.basisA(t, X)
        wB = ops.basisB(t, X)
        Ys = Vector{typeof(X)}(undef, N + 1)
        Zs = Vector{typeof(X)}(undef, N)
        Ws = Vector{typeof(X)}(undef, N)
        Ys[1] = X
        @inbounds for l in 1:N
            m = metas[l]
            W = ops.weight(p, m, wA)
            b = ops.bias(p, m, wB)
            Z = ops.affine(W, Ys[l], b)
            Ws[l] = W
            Zs[l] = Z
            Ys[l+1] = σ.(Z)
        end
        Y = Ys[N+1]
        function vjp(cotangent)
            Δp = zero(p)
            @inbounds for l in N:-1:1
                m = metas[l]
                S = activation_derivative(σ, Zs[l]) .* cotangent
                ops.dweight!(Δp, m, S, Ys[l], wA)
                ops.dbias!(Δp, m, S, wB)
                cotangent = ops.tmul(Ws[l], S)
            end
            return cotangent, Δp
        end
        return Y, vjp
    end
end

"""
    cpu_field_ops(repA, repB) → NamedTuple of primitives

Non-batched method bundle for `build_block`
"""
function cpu_field_ops(repA, repB)
    return (
        basisA=(t, X) -> device_basis_weights(repA, t, X),
        basisB=(t, X) -> device_basis_weights(repB, t, X),
        weight=(p, m, wA) -> reshape(reshape(view(p.A, m.rangeA), m.wo * m.wi, length(wA)) * wA, m.wo, m.wi),
        bias=(p, m, wB) -> reshape(view(p.B, m.rangeB), m.wo, length(wB)) * wB,
        affine=(W, Y, b) -> W * Y .+ b,
        tmul=(W, S) -> W' * S,
        (dweight!)=(Δp, m, S, Y, wA) -> (@view(Δp.A[m.rangeA]) .= vec(vec(S * Y') * wA')),
        (dbias!)=(Δp, m, S, wB) -> (@view(Δp.B[m.rangeB]) .= vec(sum(S; dims=2) * wB')),
    )
end

end
