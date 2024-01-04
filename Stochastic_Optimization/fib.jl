function fib(n)
    if n < 2
        return n
    else
        return fib(n-1) + fib(n-2)
    end
end

for i in 1:5
    @time fib(i)
end