"""
Linear interpolation helpers. Used by Spline and solve_picard.
"""
module LinearInterp

export lerp, interpolate_samples

lerp(a, b, t::Real) = @. a * (1 - t) + b * t

"""
    interpolate_samples(samples, t)

Approximate f(t) from evenly spaced samples of f on [0,1], interpolating linearly.
"""
function interpolate_samples(samples, t::Real)
    N = length(samples)
    t = clamp(t, 0.0, 1.0)
    scaled = t * (N - 1) + 1
    i = floor(Int, scaled)
    i >= N && return samples[end]
    i < 1 && return samples[1]
    return lerp(samples[i], samples[i+1], scaled - i)
end

end
