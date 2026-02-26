import Positions as pos
import time

start = time.perf_counter()
print(pos.positions['Moon'][1:4][0,0:10
                ])

end = time.perf_counter()

elapsed = end-start

print(f"{elapsed:.6f} seconds")