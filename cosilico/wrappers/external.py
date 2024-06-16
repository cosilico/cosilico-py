"""
Realtime subprocess execution
"""

from subprocess import Popen, PIPE

def realtime_subprocess(cmds, shell=False):
    with Popen(cmds, stdout=PIPE, shell=shell, encoding='utf-8') as p:
        while True:
            line = p.stdout.readline()
            if line:
                print(line)
            else:
                break
    # with Popen(cmds, stdout=PIPE, **kwargs) as p:
    #     while True:
    #         text = p.stdout.read1().decode("utf-8")
    #         print(text, end='', flush=True)


# # child_process.py
# from time import sleep

# while True:
#     # Make sure stdout writes are flushed to the stream
#     print("Spam!", end=' ', flush=True)
#     # Sleep to simulate some other work
#     sleep(1)