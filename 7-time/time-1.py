import time
from datetime import timedelta

start_time = time.time()
input()
end_time = time.time()

td = timedelta(seconds=int(round(end_time - start_time)))
print(td)