import concurrent.futures
import time

def func(i):
    time.sleep(i)
    return i*i


list = [1,2,3,6,6,6,90,100]
async_executor = concurrent.futures.ThreadPoolExecutor(2)
futures = {async_executor.submit(func, i): i for i in list}
for ii, future in enumerate(concurrent.futures.as_completed(futures)):
    print(ii, "result is", future.result())
    if ii == 2:
        async_executor.shutdown(wait=False)
        for victim in futures:
            victim.cancel()
        break