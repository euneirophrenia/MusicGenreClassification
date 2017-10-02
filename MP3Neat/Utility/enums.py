from enum import Enum


class IODirection(Enum):
    Load = 1
    Save = 2

    def __str__(self):
        return self.name.lower()



class RegistryKey(Enum):
    """ For registry access. So that you don't have to remember the exact string every time, a bit more scalable"""

    ## GENERAL KEYS ##
    BEST_NET = 'best net'
    BEST_GENOME = 'best genome'
    NET_KIND = 'kind'
    CPU_CORES = 'parallel cores'
    ALGORITHM = 'algorithm'
    OUTPUT_DIMENSION = 'spatial'
    REUSE_PREVIOUS = 'reuse past'
    CONFIGURATION = 'configuration'

    TRAIN_SCORE = 'training score'
    TRAIN_ERRORS = 'training errors'
    TRAIN_SET = 'training set'

    CONTROL_ERRORS = 'control errors'
    CONTROL_SET = 'control set'
    CONTROL_SCORE = 'control score'

    SWAP_SET = 'swapping set'
    SWAP_ERRORS = 'swapping errors'
    SWAP_SCORE = 'swap score'

    TIMESTAMP = 'timestamp'
    GENERATIONS = 'generations'


