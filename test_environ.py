import os
import threading
import time

def thread_func():
    for _ in range(1000):
        os.environ["TEST_VAR"] = "some_value"
        os.environ.pop("TEST_VAR", None)

threads = []
for _ in range(10):
    t = threading.Thread(target=thread_func)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Done")
