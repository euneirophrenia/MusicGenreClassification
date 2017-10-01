import neat
import datatools
from multiprocessing import Pool
import numpy
from NEAT import visualize
import os
import random
#from gpuacceleration import *

################################
#   DO NOT UPGRADE TO NEAT-PYTHON >= 0.9.2
#       IT CHANGES THE GENOME OBJECT SO THE SAVED GENOMES WILL NO LONGER BE DEPICKLED
#
################################

_genInterval  = 100    # for checkpoint report
_timeInterval = 300    # for check point report

def evaluateGenome(genome, config, ins, outs, recurrent=False):
    net = neat.nn.FeedForwardNetwork.create if not recurrent else neat.nn.RecurrentNetwork.create
    return standardEvaluate(net(genome, config), ins, outs)

"""def evaluateGPU(genome, config, ins, outs, recurrent=False):
    n = neat.nn.FeedForwardNetwork.create if not recurrent else neat.nn.RecurrentNetwork.create
    net = n(genome, config)
    actual = [net.activate(i) for i in ins]
    return gpuErrorEvaluate(actual, outs)"""

def standardEvaluate(net, ins, outs, func = None):
    error = 0.0
    if func is None:
        func = lambda expected, actual : numpy.linalg.norm(numpy.array(expected) - numpy.array(actual))
    for xi, xo in zip(ins, outs):
        output = net.activate(xi)
        error += func(xo , output)
    return 1.0 - (error / len(outs))


def stepActivate(x):
    if x>1:
        return 1.0
    return x if x>=0.4 else 0.0


## TODO: implement a checkpoint restore system.
## STUB: <check if there're files with the checkpoint extension>
##       <check flag to determine if run the program in "recovery" mode>
##       <restore checkpoint>

class StandardNEAT:
    def __init__(self, config_file='./config', initialState=None, recurrent=False, gpu=False, checkPoints=False):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, config_file)
        self.config= neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

        self.initialState=initialState
        self.ins=[]
        self.outs=[]
        self.labels=[]
        self.recurrent = recurrent
        self.gpu=gpu
        self.nn=neat.nn.FeedForwardNetwork.create if not recurrent else neat.nn.RecurrentNetwork.create
        self.addActivationFunction('notlinear',stepActivate)
        self.checkpoints=checkPoints

    def feed(self, datapath, fetchFromHistory=False):
        self.ins, self.outs, self.labels, self.historybest = datatools.preparedata(datapath, order=self.config.genome_config.num_outputs,
                                                                                   andGetHistoryBest=fetchFromHistory)


    def run(self, generations=300, cores=5, showDiagrams=False):
        self.population = neat.Population(self.config, initial_state=self.initialState)
        if self.historybest is not None: ##todo: probably better to use them as initial state instead of replacing but hey, it's a 1 time only price, uh?
            tomutate = random.sample(self.population.population.keys(), len(self.historybest))
            for key, x in zip(tomutate, self.historybest):
                x.key = key
                self.population.population[key]=x
            self.population.species.speciate(self.population.config, self.population.population, self.population.generation)
            print('Added',len(self.historybest), 'genomes fetched from compatible previous run.')

        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        if self.checkpoints:
            self.population.add_reporter(neat.Checkpointer(_genInterval, _timeInterval))

        if not self.gpu:
            pe = NEATParallelEvaluator(cores, evaluateGenome, self.ins, self.outs, recurrent=self.recurrent)
        else:
            raise NotImplementedError("Too early, still gotta figuring it out")
            #pe=GPUPrallelEvaluator(cores, evaluateGPU, self.ins, self.outs, recurrent=self.recurrent)

        winner = self.population.run(pe.evaluate, generations)

        self.initialState = self.population.population, self.population.species, self.population.generation

        print('\nBest genome:\n{!s}'.format(winner) + '\n')
        winner_net = self.nn(winner, self.config)
        if showDiagrams:
            visualize.draw_net(self.config, winner, True, node_names=self.labels)
            visualize.plot_stats(stats, ylog=False, view=True)

        return winner_net, winner, self.initialState

    def addActivationFunction(self, name, func):
        self.config.genome_config.add_activation(name,func)


class MTSNEAT(StandardNEAT):
    def feed(self, datapath, mutatingdatapath, fetchFromHistory=False):
        super().feed(datapath)
        self.mutatingins, self.mutatingouts, _, _ = datatools.preparedata(mutatingdatapath, order=self.config.genome_config.num_outputs,
                                                                          andGetHistoryBest=fetchFromHistory)

    def run(self, generations=300, cores=5, showDiagrams=False, dataMutationRate=0.10, dataMutationInterval=100):
        best, winner, self.initialState = super().run(generations=min(dataMutationInterval, generations),cores=cores,
                                              showDiagrams=False)
        done = dataMutationInterval
        while done < generations and done <= self.initialState[2]:
            self.ins, self.outs, self.mutatingins, self.mutatingouts = MTSNEAT.randomMutate(self.ins, self.outs,
                                                                                            self.mutatingins, self.mutatingouts,
                                                                                            rate=dataMutationRate)
            best, winner, self.initialState = super().run(generations=min(dataMutationInterval, generations - done), cores=cores,
                              showDiagrams=generations - done <= dataMutationInterval and showDiagrams)

            done += min(dataMutationInterval, generations - done)

        return best,winner, self.initialState

    @staticmethod
    def randomMutate(ins, outs, mutatingins, mutatingouts, rate=0.01):
        tomutate = int(len(ins) * rate)
        chosen = random.sample(range(0, len(ins)), tomutate)
        withwhat = random.sample(range(0, len(mutatingins)), tomutate)

        for i, j in zip(chosen, withwhat):
            ins[i], mutatingins[j] = mutatingins[j], ins[i]
            outs[i], mutatingouts[j] = mutatingouts[j], outs[i]

        return ins, outs, mutatingins, mutatingouts

class NEATParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, ins, outs, timeout=None, recurrent=False):
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(num_workers)
        self.ins=ins
        self.outs = outs
        self.recurrent=recurrent

    def __del__(self):
        self.pool.close()
        self.pool.join()

    def evaluate(self, genomes, config):
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, self.ins, self.outs, self.recurrent)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


class GPUPrallelEvaluator(NEATParallelEvaluator):
    def __init__(self, num_workers, eval_function, ins, outs, timeout=None, recurrent=False):
        super().__init__(num_workers, eval_function, ins, outs, timeout, recurrent)
        raise NotImplementedError("Don't use it yet, still gotta make it work, if you do you're gonna go OoM")

    def __del__(self):
        super().__del__()

    def evaluate(self, genomes, config):
       for genome_id, genome in genomes:
           genome.fitness = self.eval_function(genome, config, self.ins, self.outs, self.recurrent)

