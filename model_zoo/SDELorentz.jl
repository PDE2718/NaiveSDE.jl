using Pkg, Revise
Pkg.activate(".")
using NaiveSDE
using Plots, Measures, LaTeXStrings

#----------------- deterministic term f ------------------
function f(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

#----------------- stochastic term g ------------------
function g(dσ, u, p, t)
    dσ[1] = 2.0
    dσ[2] = 2.0
    dσ[3] = 2.0
end

#----------------- make the problem ------------------
prob = SDEProblem(f, g,
    [1.0, 0.0, 0.0],
    (0.0, 20.0),
    nothing, nothing
)

#----------------- solve the problem ------------------
# try different methods

dt = 10000
prob_np = wigner_process(prob, dt)
sol = solve(prob, dt, RKW2, prob_np)

#----------------- visualization ------------------
# LLxy = 25
plt_lorenz = plot([[u[i] for u in sol.u] for i ∈ 1:3]...,
    xlabel=L"x",
    ylabel=L"y",
    zlabel=L"z",
    label=nothing,
    lw = 1,
    # xlims = (-LLxy,LLxy),
    # ylims=(-LLxy, LLxy),
    size = [550,500],
    zlims = (0,50),
    aspect_ratio = 1.0,
    # xticklables = nothing,
    # xticks = 
    framestyle = :box,
    camera = (60,20),
    margins = -20mm
    # title="Lorenz attractor with noise"
)
savefig(plt_lorenz,"Lorenz.pdf")