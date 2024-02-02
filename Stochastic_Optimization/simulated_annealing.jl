import Pkg; Pkg.add("Distributions")
using Base, LinearAlgebra, Statistics, Distributions # julia 내 내장 libaray

function t_schedule(m, t_init)
    return t_init/m
end

function t_log_schedule(m, t_init)
    return t_init * log(2) / log(m + 1)
end

function t_exponential_schedule(m)
    γ = 1/4
    return m*γ
end

function simulated_annealing_vallina(f, x, k_max; T=Normal(), t=t_schedule, t_init=1)
    y = f(x) # initial point value
    x_best, y_best = x, y
    for k in 1:k_max
        x_search = x + rand(T, length(x)) # length : n-th dimension
        y_search = f(x_search)
        Δy = y_search - y # 이전점에서 찾은 y값으로 비교 best_point X
        if Δy ≤ 0 || rand() < exp(-Δy / t(k, t_init)) # metropolis criterion
            x, y = x_search, y_search
        end
        if y_search < y_best
            x_best, y_best = x_search, y_search
        end
    end
    println("T : ",t(k_max, t_init))
    return x_best, y_best
end

function simulated_annealing_bestchange_boost(f, x, k_max; T=Normal(), t=t_schedule, t_init=1)
    y = f(x) # initial point value
    m=1
    x_best, y_best = x, y
    for k in 1:k_max
        x_search = x + rand(T, length(x)) # length : n-th dimension
        y_search = f(x_search)
        Δy = y_search - y # 이전점에서 찾은 y값으로 비교 best_point X
        if Δy ≤ 0 || rand() < exp(-Δy / t(m, t_init)) # metropolis criterion
            x, y = x_search, y_search
        end
        if y_search < y_best
            m+=1
            x_best, y_best = x_search, y_search
        end
    end
    println("T : ",t(m, t_init))
    return x_best, y_best
end

function simulated_annealing_log(f, x, k_max; T=Normal(), t=t_schedule, t_init=1)
    y = f(x) # initial point value
    x_best, y_best = x, y
    for k in 1:k_max
        x_search = x + rand(T, length(x)) # length : n-th dimension
        y_search = f(x_search)
        Δy = y_search - y # 이전점에서 찾은 y값으로 비교 best_point X

        t_fn = 1e-10
        if Δy ≤ 0 || rand() < exp(-Δy / t_log_schedule(k, t_init)) # metropolis criterion
            x, y = x_search, y_search
        end
        if y_search < y_best
            x_best, y_best = x_search, y_search
        end
    end
    println("T : ",t_log_schedule(k_max, t_init))
    return x_best, y_best
end

function simulated_annealing_bestchange_exp(f, x, k_max; T=Normal(), t=t_schedule, t_init=10)
    y = f(x) # initial point value
    m=1
    x_best, y_best = x, y
    for k in 1:k_max
        x_search = x + rand(T, length(x)) # length : n-th dimension
        y_search = f(x_search)
        Δy = y_search - y # 이전점에서 찾은 y값으로 비교 best_point X
        if Δy ≤ 0 || rand() < exp(-Δy / t_exponential_schedule(m)) # metropolis criterion
            x, y = x_search, y_search
        end
        if y_search < y_best
            m+=1
            x_best, y_best = x_search, y_search
        end
    end
    println("T : ", t_exponential_schedule(m))
    return x_best, y_best
end


function corana_update!(v,a,c,ns)
    for i in eachindex(v)
        ai, ci = a[i], c[i]
        if ai > 0.6ns
            v[i] *= (1 + ci*(ai/ns-0.6)/0.4)
        elseif ai < 0.4ns
            v[i] /= (1 + ci*(0.4-ai/ns)/0.4)
        end
    end
    return v
end

function adaptive_simulated_annealing_vallina(f, x;
    v=fill(1e-5, length(x)), t=1, ϵ=1e-20, ns=20, nϵ=4, nt=max(100,5length(x)), γ=0.85,c=fill(2,length(x)))
    """
    f : objective function
    x : initial point  
    v : initial step vector
    t : initial temperature
    ϵ : stopping criterion
    ns : step vector를 줄이기 이전 cycle 수
    nt : tenoerature를 줄이기 이전 cycle 수
    """
    y = f(x)
    x_best, y_best = x, y
    y_arr, n, U = [], length(x), Uniform(-1.0,1.0)
    a, counts_cycles, counts_resets = zeros(n), 0, 0

    function basis(i, n)
        b = zeros(n)
        b[i] = 1
        return b
    end

    while true
        
        for i in 1:n
            x1 = clamp.(x + basis(i,n)*rand(U)*v[i], -1, 1)  # x1 범위 제한
            y1 = f(x1)
            Δy = y1 - y
            if Δy ≤ 0 || rand() < exp(-Δy/t) # metropolis criterion
                x, y = x1, y1
                a[i] += 1
                if y1 < y_best
                    x_best, y_best = x1, y1
                end
            end
        end
        
        counts_cycles += 1
        counts_cycles ≥ ns || continue

        counts_cycles = 0
        corana_update!(v,a,c,ns) # step vector 조정
        fill!(a,0) # a 초기화
        
        counts_resets += 1
        counts_resets ≥ nt || continue
        

        t *= γ # exponential annealing
        counts_resets
        push!(y_arr, y)
        
        if !(length(y_arr) > nϵ && y_arr[end] - y_best ≤ ϵ && all(abs(y_arr[end] - y_arr[end-u]) ≤ ϵ for u in 1:nϵ))
            x, y = x_best, y_best
        else
            break
        end

    end
    print("T : ",t)
    return x_best, y_best
end

function adaptive_simulated_annealing_boost(f, x;
    v=fill(1e-5, length(x)), t=1, ϵ=1e-20, ns=20, nϵ=4, nt=max(100,5length(x)), c=fill(2,length(x)))
    """
    f : objective function
    x : initial point  
    v : initial step vector
    t : initial temperature
    ϵ : stopping criterion
    ns : step vector를 줄이기 이전 cycle 수
    nt : tenoerature를 줄이기 이전 cycle 수
    """
    y = f(x)
    x_best, y_best = x, y
    y_arr, n, U = [], length(x), Uniform(-1.0,1.0)
    a, counts_cycles, counts_resets = zeros(n), 0, 0

    function basis(i, n)
        b = zeros(n)
        b[i] = 1
        return b
    end

    k=0
    while true
        
        for i in 1:n
            x1 = clamp.(x + basis(i,n)*rand(U)*v[i], -1e10, 1e10)  # x1 범위 제한
            y1 = f(x1)
            Δy = y1 - y
            if Δy ≤ 0 || rand() < exp(-Δy/t) # metropolis criterion
                x, y = x1, y1
                a[i] += 1
                if y1 < y_best
                    x_best, y_best = x1, y1
                end
            end
        end
        
        counts_cycles += 1
        counts_cycles ≥ ns || continue

        counts_cycles = 0
        corana_update!(v,a,c,ns) # step vector 조정
        fill!(a,0) # a 초기화
        
        counts_resets += 1
        counts_resets ≥ nt || continue
        
        k += 1
        t = t/k # fast annealing
        counts_resets
        push!(y_arr, y)
        
        if !(length(y_arr) > nϵ && y_arr[end] - y_best ≤ ϵ && all(abs(y_arr[end] - y_arr[end-u]) ≤ ϵ for u in 1:nϵ))
            x, y = x_best, y_best
        else
            break
        end

    end
    print("T : ",t)
    return x_best, y_best
end

## test
function ackley(x, a=20, b=0.2, c=2π)
    d = length(x)
    return -a*exp(-b*sqrt(sum(x.^2)/d)) - exp(sum(cos.(c*xi) for xi in x)/d) + a + exp(1)
end

f = ackley
x = [1,1]
k_max = 10^7
t_init = 0.0001

@time sa1 = simulated_annealing_vallina(f, x, k_max, t_init=t_init)
@time sa2 = simulated_annealing_bestchange_boost(f, x, k_max, t_init=t_init)
@time sa3 = simulated_annealing_log(f, x, k_max, t_init=t_init)
@time sa4 = simulated_annealing_bestchange_exp(f, x, k_max, t_init=t_init)
@time as1 = adaptive_simulated_annealing_vallina(f, x)#, t=t_init)
@time as2 = adaptive_simulated_annealing_boost(f, x)#, t=t_init)



println("vallina + boost : ",sa1)
println("vallina + log : ",sa3)
println("best change + boost : ",sa2)
println("best change + exp : ",sa4)

println("adaptive + exp : ", as1)
println("adaptive + boost : ", as2)
