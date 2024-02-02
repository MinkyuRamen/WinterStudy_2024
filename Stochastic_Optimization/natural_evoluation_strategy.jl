using Distributions
using LinearAlgebra

function natural_evolution_strategies(f, θ; k_max=1000, m=100, α=1e-3)
    μ, Σ = θ
    dimension = length(μ)
    for k in 1 : k_max

        # 샘플 및 그래디언트 초기화
        samples = Matrix{Float64}(undef, dimension, m)
        ∇μ = zeros(dimension)
        ∇Σ = I * dimension

        # 분포에서 샘플링 및 그래디언트 계산
        for i in 1:m
            sample = rand(MvNormal(μ, Symmetric(Σ)))
            samples[:, i] = sample
            score = f(sample)
            ∇μ += score * (sample - μ)
            ∇Σ += score * (sample - μ) * (sample - μ)'
        end

        # 평균과 공분산 행렬 업데이트
        ϵ = 1e-5
        μ += α * ∇μ / m
        Σ += α * ∇Σ / m + ϵ * I
    end

    return μ, Σ
end


function quadratic(x)
    return (x[1] - 1.562)^2 + (x[2] + 3.272)^2
end

f = quadratic

dimension = 2
μ = zeros(dimension)
Σ = Matrix(I, dimension, dimension)
θ = μ, Σ
k_max = 10000000000 # 분산이 매우커서 convergence가 느리다.

θ1 = natural_evolution_strategies(f, θ, k_max=k_max)
println("μ : ",θ1[1])
println("Σ : ",θ1[2])
