using Pkg, Revise
Pkg.activate(".")
using NaiveSDE
using LinearAlgebra, Statistics
using Plots, Measures, LaTeXStrings

p = (α=1.,β=1.)
u_analytic(u0, t, W) = u0 .* exp.((p[:α] - (p[:β]^2) / 2)*t .+ p[:β].*W)

function f(du, u, p, t)
    du .= p[:α] .* u
end
function g(dσ, u, p, t)
    dσ .= p[:β] .* u
end
function f_s(du, u, p, t)
    du .= (p[:α] - p[:β]^2/2) .* u 
end

dt = 200
tspan = (0.0, 2.0)
u0 = [1.0,]

prob = SDEProblem(f_s, g, u0, tspan, p, p)

# wprocs = wigner_process(prob,dt)

sol = solve(prob, dt, trivialEM, wprocs)
u_ana = [u_analytic(u0,t,W) for (t,W) in zip(sol.t,sol.W)]
sol_err = 10abs.([u[1] for u ∈ sol.u] .- [u[1] for u ∈ u_ana])

plt1 = plot(sol.t, [u[1] for u ∈ u_ana], c = 4, lw = 1,
    xlabel = L"t",
    ylabel = L"u(t)",
    xlims = (0.0,2.0),
    ylims = (0.0,40.0),
    label = "analytical"
)
plot!(sol.t,[u[1] for u ∈ sol.u], c = 1, lw = 1, l=:dashdot,
    ribbon=sol_err, fill_alpha = 0.2, label = "numerical-EM"
)

plt0 = plot(sol.t, [u[1] for u ∈ u_ana], c=4, lw=1,
    # xlabel=L"t",
    # ylabel=L"u(t)",
    xlims=(0.0, 2.0),
    ylims=(0.0, 40.0),
    label=nothing,
    xticks = nothing,
    yticks = nothing,
    framestyle = :none,
)
plot!(sol.t, [u[1] for u ∈ sol.u], c=1, lw=1, l=:dashdot,
    ribbon=3sol_err, fill_alpha=0.2, label=nothing
)

savefig(plt0, "plt0.pdf")

