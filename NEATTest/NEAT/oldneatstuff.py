import os
import neat
import NEAT.visualize as visualize
import random


#TODO: COMPLETE REWORK TO PREPARE A SET OF OPTIONS (ex: MTSNEAT and Spatial) so that each functions works out of the box
#TODO: refactor the evaluation function to accept said options set and process it

#TODO MTSNEAT: refactor current version so that the function signature is clearer (move parameters to the config file)

######## REWORKED IN NEATCORE.PY ###############


neuralnetworks = {'feedforward' : neat.nn.FeedForwardNetwork.create,
                  'recurrent' : neat.nn.RecurrentNetwork.create}

def eval_net(net, ins, outs):
    error=0.0
    for xi, xo in zip(ins, outs):
        output = net.activate(xi)
        error += abs(output[0] - xo)
    return 1.0 - (error / len(outs))

def eval_genome(genome, config, ins, outs, kind):
    #net = neat.nn.FeedForwardNetwork.create(genome, config)
    return eval_net(neuralnetworks[kind](genome, config), ins, outs)


def mismatches(net, ins, outs, threshold=0.2):
    i = 0
    res=[]
    for xi, xo in zip(ins, outs):
        output = net.activate(xi)
        if abs(output[0] - xo) >= threshold:
            res.append((i,xo,output[0]))
            #print("Mismatched input #{!r}, expected output {!r}, got {!r}".format(i, xo, output))
        i += 1
    return res


def run(ins, outs, config_file, generations, kind, cores=4, labels=None, initial_state=None, showDiagrams=False):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    ###register here custom activation function (which must take 1 argument (the output)
    #config.genome_config.add_activation('name',func)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config, initial_state=initial_state)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to <gen> generations.
    pe = neat.ParallelEvaluator(cores, eval_genome, kind, inputs=ins, outputs=outs)
    winner = p.run(pe.evaluate, generations)
    final_state = p.population, p.species, p.generation
    #winner = p.run(eval_genome, generations)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner)+'\n')

    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_net = neuralnetworks[kind](winner, config)

    if showDiagrams:
        node_names = labels #{-1:'A', -2: 'B', 0:'Output', 1:'First hidden'}
        visualize.draw_net(config, winner, True, node_names = node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        #visualize.plot_species(stats, view=True)

    return winner_net, final_state


def start_neat(inputs, outputs, configfile="config_linear", kind="feedforward", generations=300, cores=4, labels=None, showDiagrams=True):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configfile)
    return run(inputs, outputs, config_path, generations, kind.lower(), cores=cores, labels=labels, showDiagrams=showDiagrams)


def randomMutate(ins, outs, mutatingins, mutatingouts, rate=0.01):
    tomutate = int(len(ins) * rate)
    chosen = random.sample(range(0, len(ins)), tomutate)
    withwhat = random.sample(range(0, len(mutatingins)), tomutate)

    for i, j in zip(chosen, withwhat):
        ins[i] , mutatingins[j] = mutatingins[j], ins[i]
        outs[i], mutatingouts[j] = mutatingouts[j], outs[i]

    return ins, outs, mutatingins, mutatingouts

def MTSNEAT(ins, outs, mutatingins, mutatingouts, configfile='config_linear', kind='feedforward', generations=300, cores=4, labels=None,
            dataMutationRate=0.01, dataMutationInterval=100):
    '''dataMutationRate : percentuale dell'input da sostituire a ogni <dataMutationInterval> generazioni.'''
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, configfile)
    best, state = run (ins, outs, config_path, kind=kind.lower(), generations=min(dataMutationInterval, generations),
                             cores=cores, labels=labels, showDiagrams=False)
    generations-=dataMutationInterval
    while generations > 0:
        ins, outs, mutatingins, mutatingouts = randomMutate(ins, outs, mutatingins, mutatingouts, rate=dataMutationRate)
        best, state=run(ins, outs, config_path, min(dataMutationInterval, generations), kind, cores=cores,
                        labels=labels, initial_state=state, showDiagrams= generations<=dataMutationInterval)
        generations-=min(dataMutationInterval, generations)

    return best, state