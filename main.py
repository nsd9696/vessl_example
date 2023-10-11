import vessl
import time

cnt = 0
for i in range(20):
    vessl.log(step=cnt, payload={"accuracy":cnt})
    cnt += 1
    time.sleep(20)
    print(cnt)