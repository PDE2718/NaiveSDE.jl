function filldw!(dw::AbstractArray, dt::Real)
    randn!(dw)
    dw .*= √(dt)
end

#----------------- trivialEM ------------------
export trivialEM

"""
Euler-Maruyama Method, stong order 1/2
"""
function trivialEM(s::SDEIntegrator,p::SDEProblem)
    # filldw!(s.dw, s.dt)
    p.f(s.du, s.ut, p.pu, s.t)
    p.g(s.dσ, s.ut, p.pσ, s.t)
    s.un .= s.ut .+ s.du .* s.dt .+ s.dσ .* s.dw
    return nothing
end
trivialEM(u0::AbstractArray) = nothing



#----------------- EulerHeun ------------------
export EulerHeun

"""
EulerHeun Method, stong order 1/2 in Stratonovich sense.
"""
function EulerHeun(s::SDEIntegrator, p::SDEProblem)
    t, dt = s.t, s.dt
    # filldw!(s.dw, s.dt)
    ũ, dũ, dσ̃ = s.bf
    p.f(s.du, s.ut, p.pu, t)
    p.g(s.dσ, s.ut, p.pσ, t)
    ũ .= @. s.ut + s.du * dt + s.dσ * s.dw
    p.f(dũ, ũ, p.pu, t + dt)
    p.g(dσ̃, ũ, p.pσ, t + dt)
    s.un .= @. s.ut + (s.du + dũ) * (s.dt / 2) + (s.dσ + dσ̃) * (s.dw / 2)
    return nothing
end
EulerHeun(u0::AbstractArray) = [similar(u0) for _ ∈ 1:3]


#----------------- RK1 ------------------
export RK1

"""
Runge-Kutta Method (gradient free Milstein), stong order 1.
"""
function RK1(s::SDEIntegrator, p::SDEProblem)
    t, dt, Δ = s.t, s.dt, √s.dt
    # filldw!(s.dw, dt)

    p.f(s.du, s.ut, p.pu, t)
    p.g(s.dσ, s.ut, p.pσ, t)
    s.un .= @. s.ut + s.du * dt + s.dσ * s.dw
    
    p.g(s.bf, s.ut + s.dσ .* √dt, p.pσ, t)
    s.un .+= @. (1/2Δ) * (s.bf-s.dσ) * (abs2(s.dw)-dt)
    return nothing
end
RK1(u0::AbstractArray) = similar(u0)

#----------------- RK1 ------------------
export RKW2

"""
Runge-Kutta of weak order 2. Platen (1987)
"""
function RKW2(s::SDEIntegrator, p::SDEProblem)
    t, dt = s.t, s.dt
    Δ = √dt
    # filldw!(s.dw, dt)
    uc, ua, ub, duc, dσa, dσb = s.bf

    p.f(s.du, s.ut, p.pu, t)
    p.g(s.dσ, s.ut, p.pσ, t)

    uc .= @. s.ut + s.du * dt
    ua .= uc .+ s.dσ .* Δ
    ub .= uc .- s.dσ .* Δ
    uc .+= s.dσ .* s.dw
    p.f(duc, uc, p.pu, t + dt)
    p.g(dσa, ua, p.pσ, t + dt)
    p.g(dσb, ub, p.pσ, t + dt)

    s.un .= @. s.ut + (1 / 2) * (duc + s.du) * dt + (1 / 4) * (dσa + dσb + 2s.dσ) * s.dw + (1 / 4Δ) * (dσa - dσb) * (abs2(s.dw) - dt)

    return nothing
end
RKW2(u0::AbstractArray) = [similar(u0) for i ∈ 1:6]