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


def correct(basename, parent):
    tokens = basename.split('.')
    tokens.insert(len(tokens)-1, os.path.basename(parent)+".")
    return ''.join(tokens)


def flatten(basefolder, andClean=False):
    for f in os.listdir(basefolder):
        current = os.path.join(basefolder, f)
        if os.path.isdir(current):
            for x in os.listdir(current):
                if os.path.isfile(os.path.join(current,x)):
                    try:
                        os.rename(os.path.join(current,x), os.path.join(basefolder, os.path.basename(x)))
                    except FileExistsError:
                        os.rename(os.path.join(current, x), os.path.join(basefolder, correct(os.path.basename(x), current)))
                if os.path.isdir(os.path.join(current, x)):
                    flatten(current, andClean=andClean)
    if andClean:
        clean(basefolder)

def clean(basefolder):
    for f in os.listdir(basefolder):
        current = os.path.join(basefolder, f)
        if os.path.isdir(current) and len(os.listdir(current))==0:
            os.rmdir(current)


def aggregate(lista):
    res={}
    for x in lista:
        if x in res:
            res[x]+=1
        else:
            res[x]=1
    return res