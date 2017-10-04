import datatools
import numpy
from NEAT.neatcore import *
from statistics import *
import itertools
import mixed_utils
import collections
import os
import pandas
import math

algorithms = ['NEAT standard', 'Mutating Training Set NEAT']
spatials = [1, 2, 3]
datasets = ['./Datasets/MIDI/'+file for file in os.listdir('./Datasets/MIDI') if '_' not in file]



def printHistoryValues(tags):
    rows = [[h[tag] for tag in tags] for h in history]
    res=pandas.DataFrame(rows) #overkill, but yeah
    print(res)



def matches(fslist):
    return [(x,y, len(set(fslist[x]) & set(fslist[y]))) for x in fslist for y in fslist if x!=y]


def showHistoryErrors():
    for h in history:
        print('\n\n----',h['training set']['path'], '(spatial:',h['spatial'],')')
        ins, outs, l, _ = datatools.preparedata(h['training set']['path'], h['spatial'])
        datatools._showerrors(h['training set']['path'], ins, outs, 'train', h['best net'])

def plotranks():
    # for d,a,s in itertools.product(datasets, algorithms, spatials):
    #    print(d,a,'(spatial:',s,') ',len([x for x in history if x['algorithm']==a and x['spatial']==s and x['training set']['path']==d]))

    """for d in [x for x in history if x['training set']['path']==datasets[1] and x['spatial']==2]:
        print('\n\n################################', d['generations'],'algorithm:', d['algorithm'])
        datatools.showErrors(d)

    exit(0)"""

    for d, a, s in itertools.product(datasets, algorithms, spatials):
        values = [x[RegistryKey.CONTROL_SCORE] for x in history if x[RegistryKey.ALGORITHM] == a and
                  x[RegistryKey.OUTPUT_DIMENSION] == s and datatools.DataManager().compatible(datatools.DataManager().metadata(d), x[RegistryKey.TRAIN_SET])]
        if len(values) > 0:
            print(d, a, s, ' \t', len(values), ':', averagePerformance(d, a, s), sum(values) / len(values))

    for x in itertools.product(datasets, algorithms, spatials):
        plotRank(x[0], x[1], x[2])
        # showrank(x[0],x[1],x[2])

        # for d in datasets:
        #   x = bestPerformer(d)
        #   print(d,':',x[0],x[1]['algorithm'], '(spatial:',x[1]['spatial'],')', x[1]['generations'], x[1]['control score'], len(x[1]['control errors']))


def testClassify(files=['./MIDI/Jazz/route_66_gw.mid', './MIDI/ClassicMusic/furelis.mid',
                                              './MIDI/Rock/1323.mid'], register='./register.dat'):
    res = datatools.MIDIExtractor().classify(files,
                                             orderSelectionCriterium=max,
                                             runEvaluationCriterium=lambda h: 1 - (
                                             len(h['control errors']) / h['control set']['size']),
                                             register=register)
    print(res)
    return res

def geneticSurgery(what, con):
    import copy
    import neat

    culprit = datamanager.get('./culprit.dat')
    new = copy.deepcopy(culprit[0]['best genome'])
    old = culprit[0]['best genome']
    for da, a in old.connections:
        if a == what:
            del new.connections[(da, a)]
            new.connections[(da,con)]=neat.DefaultGenome.create_connection(culprit[0]['configuration'].genome_config,da,con)
        elif a == con:
            del new.connections[(da, a)]
            new.connections[(da, what)] = neat.DefaultGenome.create_connection(culprit[0]['configuration'].genome_config, da, what)
    newnet = neat.nn.FeedForwardNetwork.create(new, culprit[0]['configuration'])
    culprit[0]['best genome'] = new
    culprit[0]['best net'] = newnet

    return culprit[0]

def performSurgery():
    supposed = {'./MIDI/Jazz/route_66_gw.mid': 'jazz', './MIDI/ClassicMusic/furelis.mid': 'classic',
                './MIDI/Rock/1323.mid': 'rock'}

    culprit_ts = datamanager.get('./culprit.dat')[0]['timestamp']
    newone = geneticSurgery(0, 0) #try all combinations, also (0,0) for some reason still alters shit, look into it

    for h in history:
        if h['timestamp'] == culprit_ts:
            h = newone
        datamanager._savePickle(h, './fixedall.dat')

    res = testClassify(files = supposed.keys(), register = './fixedall.dat')
    if all(supposed[key] == res[0][os.path.basename(key)][0] for key in supposed):
        print(da, a, 'solve the problem.')
    else:
        os.remove('./fixedall.dat')

class Testing:
    def __init__(self, **kwargs):
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

def counting(raw):
    count = {}
    count_combo = {}
    for d in raw:
        keycombo = '#'.join(sorted(d['genre']))
        if keycombo in count_combo:
            count_combo[keycombo] += 1
        else:
            count_combo[keycombo] = 1
        for key in d['genre']:
            if not key in count:
                count[key] = 1
            else:
                count[key] += 1
    return count, count_combo

def partition():
    mappp = {}
    for x in counting(filtered_combo)[1]:
        mappp[x] = [f for f in filtered_combo if x == '#'.join(sorted(f['genre']))]

    toTrain = []
    toControl = []
    toSwap = []
    for genre in mappp:
        toTrain.extend(random.sample(mappp[genre], 250))
        toControl.extend(random.sample([x for x in mappp[genre] if x not in toTrain], 50))
        toSwap.extend(random.sample([x for x in mappp[genre] if x not in toTrain and x not in toControl], 50))

    for x in toTrain:
        assert x not in toControl
        assert x not in toSwap
        if 'Soundtrack' in x['genre']:
            x['genre'] = ['Instrumental']
        datamanager._savePickle(x, './Datasets/MP3/training.dat')

    for x in toControl:
        assert x not in toSwap
        assert x not in toTrain
        if 'Soundtrack' in x['genre']:
            x['genre'] = ['Instrumental']
        datamanager._savePickle(x, './Datasets/MP3/control.dat')
    for x in toSwap:
        assert x not in toControl
        assert x not in toTrain
        if 'Soundtrack' in x['genre']:
            x['genre'] = ['Instrumental']
        datamanager._savePickle(x, './Datasets/MP3/swap.dat')


if __name__=='__main__':
    raw, meta = datamanager.get('./Datasets/MP3/all.dat', andGetMeta=True)

    train = datamanager.get('./Datasets/MP3/train.arff')
    control = datamanager.get('./Datasets/MP3/control.arff')
    swap = datamanager.get('./Datasets/MP3/swap.arff')
    compr = datamanager.get('./Datasets/MP3/compressed.pickle')
    print(len(train), len(control), len(swap), len(compr), len(raw))

    #print(len([x for x in train if any(math.isnan(x[k]) for k in x if type(x[k])==float)]))

    res=datamanager.get('./Datasets/MP3/filtered.dat')
    processed = {x['title']: x for x in res}
    #datamanager.save(res, './test.arff')
    lols=datamanager.get('./test.arff')
    print(lols[0]['title'], lols[0]['genre'])

##todo:: create pickle files from arffs processing genre -> genre.split('/')











