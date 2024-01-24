import Pkg; Pkg.add("Distributions")
using Base, LinearAlgebra, Statistics, Distributions # julia 내 내장 libaray

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
t_init = 1


@time as1 = adaptive_simulated_annealing_vallina(f, x)#, t=t_init)
@time as2 = adaptive_simulated_annealing_boost(f, x)#, t=t_init)


println("adaptive + exp : ", as1)
println("adaptive + boost : ", as2)
