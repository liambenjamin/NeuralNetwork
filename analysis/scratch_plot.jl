include("../src/neurons.jl")
include("../src/network.jl")
include("../src/partition.jl")
include("../src/rnn.jl")
include("../src/gru.jl")
include("../src/lstm.jl")
include("../src/helpers.jl")
include("../src/penalties.jl")
include("plot.jl")


using FileIO, DataFrames, CSV, LaTeXStrings, ImageCore
#=
lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
N = 2500
feats, labels = Helpers.load_mnist_test()
method = "adjoint"
ntwk = load("mnist3/rnn_adjoint_5.jld2", "ntwk")
partition = load("mnist3/rnn_partition.jld2", "partition")
pos = 1
ϵ = 0.5

X, Π = Helpers.perturb_images(ntwk, lsmod, pnmod, N, feats, labels, partition, method, pos, 1.0)
X2, Π2 = Helpers.perturb_images(ntwk, lsmod, pnmod, N, feats, labels, partition, method, pos, 0.5)
X3, Π3 = Helpers.perturb_images(ntwk, lsmod, pnmod, N, feats, labels, partition, method, pos, 0.1)

image_6 = X[12]
save("mnist3/image_6.png", image_6)


"""
Linear Interpolation between adjoint and coadjoint solutions (mnist)
"""

adj_ntwk = load("mnist3/rnn_adjoint_5.jld2", "ntwk")
coadj_ntwk = load("mnist3/rnn_coadjoint_1e2_5.jld2", "ntwk")
lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
α = collect(-0.25:0.025:1.25)
F, E = Plot.linear_interpolation(adj_ntwk, coadj_ntwk, lsmod, pnmod, "rnn", α)

x = collect(0:0.025:1.0)

interp_plot = plot(x,E,title="Linear Interpolation:\nAdjoint & Coadjoint Solutions",
    xlabel=L"\alpha", ylabel="Test Error",labels="")
scatter!([0], [E[1]], ms=5, color=[:red],
        labels="Adjoint",legend=:topleft)
scatter!([1], [E[end]], ms=5, color=[:blue],
        labels="Coadjoint",legend=:topleft)



"""
Generating Surface Plots (MNIST)
"""

adj_ntwk = load("mnist3/rnn_adjoint_5.jld2", "ntwk")
coadj_ntwk = load("mnist3/rnn_coadjoint_1e2_5.jld2", "ntwk")
rand_ntwk1 = load("mnist3/rnn_rand_init1.jld2", "ntwk")
rand_ntwk2 = load("mnist3/rnn_rand_init2.jld2", "ntwk")



lsmod = loss.softmaxCrossEntropy
pnmod = penalty.var_phi
ntwk_type = "rnn"
dataset = "mnist"
N_samp = 500 # use for testing
# interpolate over mesh -0.25:0.025:1.25
F, E = Plot.fourPt_interpolation(adj_ntwk, rand_ntwk1, rand_ntwk2, coadj_ntwk, lsmod, pnmod, ntwk_type, dataset, N_samp)
α = β = collect(-0.25:0.05:1.25)

CSV.write("mnist3/F.csv",  DataFrame(F), writeheader=false)
CSV.write("mnist3/E.csv",  DataFrame(E), writeheader=false)
CSV.write("mnist3/mesh.csv",  DataFrame(hcat(α,β)), writeheader=false)

"""
Bilinear Interpolation Surface b/t (adjoint,coadjoint,rand1, rand2)
"""
using PyPlot
using LinearAlgebra
#using3D() # Needed to create a 3D subplot

function gridDependent(E)
    n = length(E)
    z = zeros(n,n)
    for i=1:n
        for j=1:n
            z[i,j] = E[i][j]
        end
    end
    return z
end

#F = CSV.File("toy/F.csv")
F = CSV.File("toy/Fr1.csv")
mesh = CSV.File("toy/mesh.csv")
α = β = map(i -> mesh[i][1], 1:length(mesh))

xgrid = repeat(α',length(α),1)
ygrid = repeat(β,1,length(β))
z = gridDependent(F)

fig = figure("pyplot_surfaceplot",figsize=(10,10))
#ax = fig.add_subplot(2,1,1,projection="3d")
plot_surface(xgrid, ygrid, z, rstride=2,edgecolors="k", cstride=2,
            cmap=ColorMap("gray"), alpha=0.6, linewidth=0.25,
            )
xlabel("α")
ylabel("β")
zlabel("Test Error")

PyPlot.title("Adjoint-Coadjoint Interpolation")
=#
"""
Plotting Perturbed Test Error(s)
"""


using Plots

"""
Adjoint first 2 principal directions perturbed error
"""

name = ["5e4", "1e3", "5e3", "1e2","5e2", "1e1", "2.5e1", "5e1", "7.5e1", "1e0"]

adj_E1, co_E1, adj_F1, coadj_F1 = Plot.unpackNtwkInfo(name, "u1", 1)
adj_E14, co_E14, adj_F14, coadj_F14 = Plot.unpackNtwkInfo(name, "u14", 1)
adj_E28, co_E28, adj_F28, coadj_F28 = Plot.unpackNtwkInfo(name, "u28", 1)

d2_adj_E1, d2_co_E1, d2_adj_F1, d2_coadj_F1 = Plot.unpackNtwkInfo(name, "u1",2)
d2_adj_E14, d2_co_E14, d2_adj_F14, d2_coadj_F14 = Plot.unpackNtwkInfo(name, "u14",2)
d2_adj_E28, d2_co_E28, d2_adj_F28, d2_coadj_F28 = Plot.unpackNtwkInfo(name, "u28",2)


x = [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 7.5e-1, 1.0]
x_names = ["5e-4", "1e-3", "5e-3", "1e-2", "5e-2", "1e-1", "2.5e-1", "5e-1", "7.5e-1", "1.0"]

adj_pE1 = adj_E1[2:end] .- adj_E1[1]
adj_pE7 = adj_E7[2:end] .- adj_E7[1]
adj_pE28 = adj_E28[2:end] .- adj_E28[1]

d2_adj_pE1 = d2_adj_E1[2:end] .- d2_adj_E1[1]
d2_adj_pE14 = d2_adj_E14[2:end] .- d2_adj_E14[1]
d2_adj_pE28 = d2_adj_E28[2:end] .- d2_adj_E28[1]
#=
adj_Π = norm.(hcat(adj_pE1, adj_pE14, adj_pE28, d2_adj_pE1, d2_adj_pE14, d2_adj_pE28))

plot(x, adj_Π, labels=[L"d_{1}(u_{1})" L"d_{1}(u_{14})" L"d_{1}(u_{28})" L"d_{2}(u_{1})" L"d_{2}(u_{14})" L"d_{2}(u_{28})"],
        legend=:topleft,
        marker = [:circle :diamond :rect :circle :diamond :rect],
        linestyle = [:solid :solid :solid :dot :dot :dot]
        xlabel=L"\epsilon",
        ylabel="Test Error Addition",
        title="Perturbed Test Error")

# end adjoint plot


p_adj = plot(x, adj_Π, layout = 1,
        markershape=[:circle :diamond :star4],
        markercolor=[:match :match :match],
        legend=[:topleft :none :none],
        label=[L"u_{1}" L"u_{14}" L"u_{28}"],
        title="Perturbed Test Error",
        xlabel=L"\epsilon\nabla u_{i}",
        ylabel="Additional Test Error"
        )
savefig(p_adj,"mnist3/pert_adjoint_.png")

co_pE1 = co_E1[2:end] .- co_E1[1]
co_pE7 = co_E7[2:end] .- co_E7[1]
co_pE14 = co_E14[2:end] .- co_E14[1]
co_pE21 = co_E21[2:end] .- co_E21[1]
co_pE28 = co_E28[2:end] .- co_E28[1]

p_keep = plot(x, hcat(adj_pE1, co_pE1, adj_pE28, co_pE28), layout = 1,
        markershape=[:circle :diamond :circle :diamond],
        markercolor=[:match :match :match :match],
        linestyle=[:solid :dash :solid :dash],
        xaxis=:log,
        legend=[:topleft :none :none],
        label=[L"Adjoint\ u_{1}" L"Coadjoint\ u_{1}" L"Adjoint\ u_{28}" L"Coadjoint\ u_{28}"],
        title="Perturbed Test Error",
        xlabel=L"\epsilon\nabla u_{i}",
        ylabel="Additional Test Error"
        )
plot!(xticks = (x,x_names), xrotation=45)

savefig(p_keep,"mnist3/pert_adjoint_coadjoint.png")

co_pE1 = co_E1[2:end] .- co_E1[1]

adj_pE28 = adj_E28[2:end] .- adj_E28[1]
co_pE28 = co_E28[2:end] .- co_E28[1]

p_keep = plot(x, hcat(adj_pE1, co_pE1, adj_pE28, co_pE28), layout = 1,
        markershape=[:circle :diamond :circle :diamond],
        markercolor=[:match :match :match :match],
        linestyle=[:solid :dash :solid :dash],
        xaxis=:log,
        legend=[:topleft :none :none],
        label=[L"Adjoint\ u_{1}" L"Coadjoint\ u_{1}" L"Adjoint\ u_{28}" L"Coadjoint\ u_{28}"],
        title="Perturbed Test Error Addition",
        xlabel=L"\epsilon\nabla u_{i}",
        ylabel="Additional Test Error"
        )
plot!(xticks = (x,x_names), xrotation=45)


#savefig(p_keep,"mnist3/pert_u1u28.png")
#savefig(p7,"mnist3/pert_u7.png")
#savefig(p14,"mnist3/pert_u14.png")
#savefig(p21,"mnist3/pert_u21.png")
#savefig(p28,"mnist3/pert_u28.png")
=#
