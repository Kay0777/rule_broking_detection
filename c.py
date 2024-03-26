from concurrent.futures import ThreadPoolExecutor, wait, Future
from random import randrange
import time


def task(i: int):
    time.sleep(randrange(1, 2))
    if i % 2:
        return True
    # Your task code here
    raise TypeError('Fock bitch!')


def handle_exception(future: Future):
    exception = future.exception()
    if exception:
        future.function_name
        print(f"Exception in task: {exception}")


with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(
            task, i
        ).add_done_callback(
            handle_exception
        ) for i in range(10)
    ]
    wait(futures)
