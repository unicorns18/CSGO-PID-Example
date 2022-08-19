from ctypes import windll
import numpy as math

TimeBeginPeriod = windll.winmm.timeBeginPeriod
HPSleep = windll.kernel32.Sleep
TimeEndPeriod = windll.winmm.timeEndPeriod
    
def FOV(target_move, base_len):
    actual_move = math.atan(target_move/base_len) * base_len
    return actual_move

# More accurate sleep function than python's built-in.
def millisleep(ms):
    TimeBeginPeriod(1)
    HPSleep(int(ms))
    TimeEndPeriod(1)