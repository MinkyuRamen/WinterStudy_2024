import Pkg; Pkg.add("Convex")
using Convex

function bracket_minimum(f, x=0; s=1e-2, k=2.0)
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya
        a, b = b, a
        ya, yb = yb, ya
        s = -s # 반대방향으로 움직인다.
    end
    while true
        c, yc = b + s, f(b + s)
        if yc > yb # c와 b 사이에 local minimum이 존재한다.
            # 조건이 true이면 ? 뒤의 값(a,c) 이 반환되고, false이면 : 뒤의 값(c,a)이 반환
            return a < c ? (a, c) : (c, a)
        end
        a, ya, b, yb = b, yb, c, yc # 값을 업데이트한 후
        s *= k # learning rate를 키운다.
    end
end

## minimize를 golden_search로 파악
function golden_section_search(objective, a, b, n=10000)
    ϕ = 1.618
    ρ = ϕ - 1
    d = ρ*b + (1-ρ)*a
    yd = objective(d)
    for i = 1 : n-1
        c = ρ*a + (1-ρ)*b
        yc = objective(c)
        if yc < yd
            b, d, yd = d, c, yc
        else
            a, b = b, c
        end
    end
    return (a+b)/2
end

function line_search(f, x, d) # x:현재위치 | d : 검색방향
    objective = α -> f(x + α*d)
    a, b = bracket_minimum(objective) # local min가 있을 법한 range 추정
    α = golden_section_search(objective, a, b) # 구간안에 있는 함수의 최솟값(minimize) 추정
    return x + α*d # x에서 d방향으로 α만큼 이동
end

function basis(i,n)
    return [k==i ? 1.0 : 0.0 for k in 1:n]
end

@time function cyclic_coordinate_descent(f, x, ϵ=1e-3)
    Δ, n = Inf, length(x)
    while abs(Δ) > ϵ
        x1 = copy(x)
        for i in 1:n
            d = basis(i, n)
            x = line_search(f, x, d)
        end
        Δ = norm(x - x1)
    end
    return x
end

@time function cyclic_coordinate_descent_with_acceleration_step(f, x, ϵ=0.02)
    Δ, n = Inf, length(x)
    while abs(Δ) > ϵ
        x1 = copy(x)
        for i in 1:n
            d = basis(i, n)
            x = line_search(f, x, d)
        end
        x = line_search(f, x, x-x1) # acceleration_step
        Δ = norm(x-x1)
    end
    return x
end



#### test code
function quadratic(x)
    return (x[1] - 1.562)^2 + (x[2] + 3.272)^2 + (x[3] - 2.113)^2
end

function univar(x)
    return x^2 - 6*x + 9
end

function ackley(x, a=20, b=0.2, c=2π)
    d = length(x)
    return -a*exp(-b*sqrt(sum(x.^2)/d)) - exp(sum(cos.(c*xi) for xi in x)/d) + a + exp(1)
end

f = univar
x = 0

println("bracket_minimum :",bracket_minimum(f, x))

# f = quadratic
f = ackley
x = [1,1,1]

ccd = cyclic_coordinate_descent(f, x)
ccdwas = cyclic_coordinate_descent_with_acceleration_step(f, x)

println("cyclic_coordinate_descent : ", ccd, "time : ", @elapsed ccd)
println("cyclic_coordinate_descent_with_acceleration_step : ", ccdwas, "time: ", @elapsed ccdwas)