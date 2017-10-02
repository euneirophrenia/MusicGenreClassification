from NEAT.neatcore import StandardNEAT, MTSNEAT, standardEvaluate
from Utility import datatools
import datetime
from enums import RegistryKey

##General Setup
generations = 1000                                                               # Number of generations to run
cores = 5                                                                        # Number of parallel processes
kind = 'feedforward'                                                             # feedforward or recurrent
training_set = './Datasets/MP3/training.pickle'                                  # training set path
control_set  = './Datasets/MP3/control.pickle'                                   # brand new data to test best individuals
swapping_set = './Datasets/MP3/swap.pickle'                                      # swapping set for MTSNEAT

use_MTSNEAT = False                                                              # whether to use MTS or STANDARD NEAT                                                               # If to use spatial repr
gpuacceleration = False                                                          # Don't change just yet

reuse_past_best = False                                                           # Fetch best genomes from the past

#todo:: pass correctly the history path to the prepare data. More likely, merge registers and change the 'compatible' function
##todo:: make 'compatible' check also the file type (add the filetype to the dataset meta)

saveCheckPoints = False                                                          # Switch for very long runs

algorithm = {False:'NEAT standard', True:'Mutating Training Set NEAT'}           # for register only, don't worry
config_path = 'config_mp3.neat'                                                  # configuration file (not really the path tho)

showdiagrams = False                                                             # diagrams are actually blocking

repeatTimes  = 5                                                                 # how many consecutive runs


datamanger = datatools.DataManager()
errorFetcher = datatools.ErrorFetcher()

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
                                           dataMutationRate=0.1, dataMutationInterval=100)

        ins, outs, _, _ = datamanger.preparedata(training_set, order=runner.config.genome_config.num_outputs)
        against_trained = standardEvaluate(best, ins, outs)

        print("\n\nFitness of best individual against train data: ", against_trained)
        trainerrors=errorFetcher._showerrors(training_set, ins, outs, "training", best)[0]

        if use_MTSNEAT:
            mutins, mutouts, _, _ = datamanger.preparedata(swapping_set, order=runner.config.genome_config.num_outputs)
            against_swapped = standardEvaluate(best, mutins, mutouts)
            print("\n\nFitness of best individual against swapping set: ", against_swapped)
            swaperrors=errorFetcher._showerrors(swapping_set, mutins, mutouts, "swapping", best)[0]

        ins,outs, _, _ = datamanger.preparedata(control_set, order=runner.config.genome_config.num_outputs, forceRefresh=True)
        against_new = standardEvaluate(best, ins, outs)

        print("\nFitness of best individual against brand new data: ", against_new, "\n")
        newerrors=errorFetcher._showerrors(control_set, ins, outs, "new", best)[0]

        datatools.DataManager.save({RegistryKey.BEST_NET:best, RegistryKey.TRAIN_SCORE:against_trained, RegistryKey.CONTROL_SCORE:against_new,
                 RegistryKey.TRAIN_ERRORS:len(trainerrors), RegistryKey.CONTROL_ERRORS:len(newerrors), RegistryKey.GENERATIONS:generations,
                 RegistryKey.NET_KIND:kind, RegistryKey.CPU_CORES:cores, RegistryKey.TRAIN_SET:datamanger.metadata(training_set),
                 RegistryKey.CONTROL_SET: datamanger.metadata(control_set), RegistryKey.TIMESTAMP:datetime.datetime.now(),
                 RegistryKey.ALGORITHM: algorithm[use_MTSNEAT], RegistryKey.SWAP_SET:datamanger.metadata(swapping_set,forceRefresh=True) if use_MTSNEAT else None,
               RegistryKey.SWAP_ERRORS:len(swaperrors) if use_MTSNEAT else None, RegistryKey.SWAP_SCORE:against_swapped if use_MTSNEAT else None,
               RegistryKey.OUTPUT_DIMENSION:runner.config.genome_config.num_outputs, RegistryKey.CONFIGURATION:runner.config,
               RegistryKey.BEST_GENOME:bestgenome, RegistryKey.REUSE_PREVIOUS:reuse_past_best}, outputfile='./register_mp3.dat')


### todo:: MAKE THE HISTORY KEYS A ENUM SO THAT I CAN'T GET IT WRONG
### also, don't save ALL the errors, just save the number, it doesn't matter anymore the specific file