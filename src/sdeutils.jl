
export SDEProblem
struct SDEProblem
    f::Function
    g::Function
    u0::AbstractArray
    # w0::AbstractArray
    tspan::NTuple{2,Real}
    pu::Union{Tuple,NamedTuple,Nothing}
    pσ::Union{Tuple,NamedTuple,Nothing}
end

mutable struct SDEIntegrator
    ut::AbstractArray
    un::AbstractArray
    du::AbstractArray
    dσ::AbstractArray
    dw::AbstractArray
    bf::Union{AbstractArray,Nothing}
    t::Real
    dt::Real
end

function SDEIntegrator(u0::AbstractArray, t0, dt0)
    return SDEIntegrator(
        deepcopy(u0),
        deepcopy(u0),
        deepcopy(u0),
        deepcopy(u0),
        deepcopy(u0),
        nothing,
        t0,dt0
        )
end

export SDESolution
mutable struct SDESolution
    u::AbstractArray
    t::AbstractArray
    dw::Union{AbstractArray,Nothing}
    W::Union{AbstractArray,Nothing}
end

include("sdesteppers.jl")

function parse_tgrid(dt::Real, tspan::NTuple{2,Real})
    if dt isa Integer
        tgrid = LinRange(tspan..., dt) |> collect
    else
        tgrid = tspan[1]:dt:tspan[2] |> collect
        if tgrid[end] < tspan[2]
            push!(tgrid, tspan[2])
        end
    end
    return tgrid
end

export wigner_process
function wigner_process(prob::SDEProblem, dt::Real)
    tgrid = parse_tgrid(dt, prob.tspan)
    dwi = [zero(prob.u0) for i ∈ eachindex(tgrid)]
    for i ∈ 2:length(dwi)
        Δt = tgrid[i] - tgrid[i-1]
        filldw!(dwi[i], Δt)
    end
    return dwi
end

export solve
function solve(prob::SDEProblem, dt::Real, stepper::Function, noise_process=nothing)
    tgrid = parse_tgrid(dt, prob.tspan)
    sol = SDESolution(
        [deepcopy(prob.u0) for i ∈ eachindex(tgrid)],
        tgrid,
        isnothing(noise_process) ? noise_process(prob, dt) : noise_process,
        nothing
        )
    s = SDEIntegrator(prob.u0,sol.t[1],sol.t[2]-sol.t[1])
    s.bf = stepper(prob.u0)
    for i ∈ 1:length(sol.t)-1
        s.t = sol.t[i]
        s.dt = sol.t[i+1]-sol.t[i]
        s.ut = sol.u[i]
        s.un = sol.u[i+1]
        s.dw = sol.dw[i+1]
        stepper(s,prob)
    end
    sol.W = cumsum(sol.dw)
    return sol
end
