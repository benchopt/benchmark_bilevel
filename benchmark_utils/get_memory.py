from psutil import Process
import gc


def get_memory():
    "Get memory of a process and its children."
    gc.collect()
    gc.collect()
    p = Process()
    memory = p.memory_info().rss
    for c in p.children():
        memory += c.memory_info().rss
    return memory
