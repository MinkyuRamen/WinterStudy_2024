import time

def fib(n):
    if n < 2:
        return n
    else:
        return fib(n-1) + fib(n-2)

tot_time = time.time()
for i in range(5):
    start_time = time.time()
    print(fib(30))
    end_time = time.time()
    print("Time: ", end_time - start_time)
tot_end_time = time.time()

print("Total time: ", tot_end_time - tot_time)