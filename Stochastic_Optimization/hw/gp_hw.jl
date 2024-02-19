# import Pkg; Pkg.add("Distances")
import Pkg; Pkg.add("SpecialFunctions")
using Plots
using LinearAlgebra
using Random
using Distances
using SpecialFunctions

# Dataset
X_train = reshape([0.2, 0.4, 0.5, 0.6, 0.8, 1.0], :, 1)
y_train = reshape([sin(-4.8 * x) + 0.8 * rand() + 1 for x in X_train], :, 1)
X_test = reshape(LinRange(0, 1.2, 100), :, 1)

# RBF
function rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0)
    sqdist = sum(x1.^2, dims=2) .+ sum(x2'.^2, dims=1) .- 2 .* (x1 * x2')
    return sigma_f^2 .* exp.(-0.5 / length_scale^2 .* sqdist)
end

# Matérn
function matern_kernel(X1, X2, length_scale=1.0, sigma_f=1.0; nu=1.5)
    pairwise_dists = pairwise(Euclidean(), X1' / length_scale, X2' / length_scale)

    if nu == 1.5
        factor = sqrt(3) .* pairwise_dists
        kernel = sigma_f^2 .* (1 .+ factor) .* exp.(-factor)
    elseif nu == 2.5
        factor = sqrt(5) .* pairwise_dists
        kernel = sigma_f^2 .* (1 .+ factor .+ factor.^2 / 3) .* exp.(-factor)
    else
        factor = sqrt(2 * nu) .* pairwise_dists
        kernel = sigma_f^2 .* (2^(1-nu) / gamma(nu)) .* (factor.^nu) .* besselk(nu, factor)
        kernel[isnan.(kernel)] .= 0
    end
    return kernel
end

function plot_gp(mu, cov, X, X_train=nothing, y_train=nothing; samples=[], alpha=1.0, length_scale=1.0)
    X = vec(X)
    mu = vec(mu)
    uncertainty = 1.96 .* sqrt.(diag(cov))
    
    p = plot(X, mu, ribbon=uncertainty, fillalpha=0.1, label="Mean", title="Sigma: $alpha, Length: $length_scale", titlefont = font(9))
    if X_train !== nothing
        scatter!(p, X_train, y_train, label="Training data", color=:red, markersize=4)
    end
    for (i, sample) in enumerate(samples)
        plot!(p, X, sample, lw=1, ls=:dash, label="Sample $i")
    end
    return p
end


# hyper paramter
length_scales = [1.0, 0.5, 0.25, 0.1]
alphas = [0.01, 1, 10]

plots = []
for alpha in alphas
    for length_scale in length_scales
        K = rbf_kernel(X_train, X_train, length_scale, alpha)
        K_s = rbf_kernel(X_train, X_test, length_scale, alpha)
        K_ss = rbf_kernel(X_test, X_test, length_scale, alpha)
        K_inv = inv(K + 1e-5 * I)
        
        # 예측
        mu_s = K_s' * K_inv * y_train
        cov_s = K_ss - K_s' * K_inv * K_s

        push!(plots, plot_gp(mu_s, cov_s, X_test, X_train, y_train, samples=[], alpha=alpha, length_scale=length_scale))
    end
end

# 전체 플롯 생성 및 저장
final_plot = plot(plots..., layout=(length(alphas), length(length_scales)), size=(900, 900), legend=:none)

display(final_plot)
savefig(final_plot, "/Users/minkyuramen/Desktop/WinterStudy_2024/Stochastic_Optimization/hw/GPR_with_RBF.png")

plots = []
for alpha in alphas
    for length_scale in length_scales
        K = matern_kernel(X_train, X_train, length_scale, alpha)
        K_s = matern_kernel(X_train, X_test, length_scale, alpha)
        K_ss = matern_kernel(X_test, X_test, length_scale, alpha)
        K_inv = inv(K + 1e-5 * I)
        
        # 예측
        mu_s = K_s' * K_inv * y_train
        cov_s = K_ss - K_s' * K_inv * K_s

        push!(plots, plot_gp(mu_s, cov_s, X_test, X_train, y_train, samples=[], alpha=alpha, length_scale=length_scale))
    end
end

# 전체 플롯 생성 및 저장
final_plot = plot(plots..., layout=(length(alphas), length(length_scales)), size=(900, 900), legend=:none)

display(final_plot)
savefig(final_plot, "/Users/minkyuramen/Desktop/WinterStudy_2024/Stochastic_Optimization/hw/GPR_with_matern.png")