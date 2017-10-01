from NEAT.neatcore import StandardNEAT, MTSNEAT, standardEvaluate
from Utility import datatools
import neat
import datetime

datasets = ['classic-jazz-rock','classic-rock','classic-jazz','jazz-rock']
index = 0                                                                   # which dataset to use (in the above list)

##General Setup
generations = 1000                                                           # Number of generations to run
cores = 5                                                                    # Number of parallel processes
kind = 'feedforward'                                                         # feedforward or recurrent
training_set = './Datasets/'+datasets[index]+'.dat'                          # training set path
control_set  = './Datasets/'+datasets[index]+'_test.dat'                     # brand new data to test best individuals
swapping_set = './Datasets/'+datasets[index]+'_swap.dat'                     # swapping set for MTSNEAT
use_MTSNEAT = True                                                         # whether to use MTS or STANDARD NEAT                                                               # If to use spatial repr
gpuacceleration = False                                                      # Don't change just yet

reuse_past_best = False                                                      # Fetch best genomes from the past
saveCheckPoints = False                                                      # Switch for very long runs

algorithm = {False:'NEAT standard', True:'Mutating Training Set NEAT'}       # for register only
config_path = 'config.neat'                                                  # configuration file (not really the path tho)

showdiagrams = False                                                         # diagrams are actually blocking

repeatTimes  = 5
# how many consecutive runs, be fucking careful with this


##Run and collect info (also save them to register)
if __name__ == '__main__':
    for _ in range(0, repeatTimes):
        if not use_MTSNEAT:
            runner = StandardNEAT(config_path, recurrent= kind=='recurrent', gpu=gpuacceleration, checkPoints=saveCheckPoints)
            runner.feed(training_set, fetchFromHistory=reuse_past_best)
            best, bestgenome,_ = runner.run(generations=generations, cores=cores, showDiagrams=showdiagrams)
        else:
            runner = MTSNEAT(config_path, recurrent= kind=='recurrent', gpu=gpuacceleration, checkPoints=saveCheckPoints)
            runner.feed(training_set, swapping_set, fetchFromHistory=reuse_past_best)
            best,bestgenome,_ = runner.run(generations=generations, cores=cores, showDiagrams=showdiagrams,
                                           dataMutationRate=0.1, dataMutationInterval=50)

        ins, outs, _, _ = datatools.preparedata(training_set, order=runner.config.genome_config.num_outputs)
        against_trained = standardEvaluate(best, ins, outs)

        print("\n\nFitness of best individual against train data: ", against_trained)
        trainerrors=datatools._showerrors(training_set, ins, outs, "training", best)[0]

        if use_MTSNEAT:
            mutins, mutouts, _, _ = datatools.preparedata(swapping_set, order=runner.config.genome_config.num_outputs)
            against_swapped = standardEvaluate(best, mutins, mutouts)
            print("\n\nFitness of best individual against swapping set: ", against_swapped)
            swaperrors=datatools._showerrors(swapping_set, mutins, mutouts, "swapping", best)[0]

        ins,outs, _, _ = datatools.preparedata(control_set, order=runner.config.genome_config.num_outputs)
        against_new = standardEvaluate(best, ins, outs)

        print("\nFitness of best individual against brand new data: ", against_new, "\n")
        newerrors=datatools._showerrors(control_set, ins, outs, "new", best)[0]

        datatools.register({'best net':best, 'training score':against_trained, 'control score':against_new,
                 'training errors':trainerrors, 'control errors':newerrors, 'generations':generations,
                 'kind':kind, 'parallel cores':cores, 'training set':datatools.metadata(training_set),
                 'control set': datatools.metadata(control_set,tag='test'), 'timestamp':datetime.datetime.now(),
                 'algorithm': algorithm[use_MTSNEAT], 'swapping set':datatools.metadata(swapping_set, tag='swap') if use_MTSNEAT else None,
                 'swapping errors':swaperrors if use_MTSNEAT else None, 'swap score':against_swapped if use_MTSNEAT else None,
                 'spatial':runner.config.genome_config.num_outputs, 'configuration':runner.config,
                 'best genome':bestgenome, 'reuse past':reuse_past_best})






