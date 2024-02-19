import Random: seed!
import LinearAlgebra: norm
import Pkg; Pkg.add("PDMats")

using Distributions
using LinearAlgebra
using PDMats

function fit(P, elite_samples)
    ### elite_sample 들의 평균과 공분산 계산
    # dim=2; 각 열에 대한 평균 계산, mean 함수는 1xN 형태의 행렬이므로 vec로 1차원 벡터로 변환
    μ = mean(elite_samples, dims=2) |> vec
    ϵ = 1e-5 # Σ에 작은 값을 더해서 normalize
    # 전치(')를 추가하여 각 열이 하나의 샘플을 나타내도록 함
    Σ = cov(elite_samples', corrected=false) + ϵ * I # corrected=false; 표본의 크기로 나누어 unbiasd estimator로 만들어줌
    # 양의 정부호 대칭 행렬을 만듬
    Σ = PDMat(Symmetric(Σ))

    # P_type에 해당하는 분포 객체를 생성하고, 계산된 평균과 공분산으로 초기화
    return P(μ,Σ)
end

function cross_entropy_method(f, P, k_max; m=100, m_elite=10)
    for k in 1 : k_max
        samples = rand(P, m)
        # println(samples)
        order = sortperm([f(samples[:,i]) for i in 1 : m])
        P = fit(typeof(P), samples[:,order[1 : m_elite]])
    end
    return P
end


## test code
seed!(0)
f = x->norm(x)
μ = [0.5, 1.5]
Σ = [1.0 0.2; 0.2 2.0]
P = MvNormal(μ, Σ)
k_max = 10

P = cross_entropy_method(f, P, k_max)
println("P : ",P)

function quadratic(x)
    return (x[1] - 1.562)^2 + (x[2] + 3.272)^2
end

f = quadratic
μ = [0; 0]
Σ = [1 0; 0 1]
P = MvNormal(μ, Σ)
k_max = 100

P = cross_entropy_method(f, P, k_max)
println("P : ",P)