import time
import os

def timeit(func):
    def timed(*args, **kwargs):
        start = time.clock()
        res = func(*args, **kwargs)
        print("[DEBUG] Executed ",func.__name__, " in ", 1000*(time.clock()-start), " ms")
        return res
    return timed

class deprecated:
    def __init__(self, fatal=False):
        self.fatal = fatal

    def __call__(self, f):
        def youshallnotexecute(*args, **kwars):
            if self.fatal:
                raise Exception("It's deprecated man, why do you use this")
            raise DeprecationWarning("You may regret this. Trust me, I coded that.")
            return f(*args, **kwargs)
        return youshallnotexecute




