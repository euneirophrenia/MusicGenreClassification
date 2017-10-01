import datatools
import numpy
from NEAT.neatcore import *
from statistics import *
import itertools


def getpath(meta, tag='training'):
    if meta['path'] != '':
        return meta['path']

    datasets = ['classic-jazz-rock', 'classic-rock', 'classic-jazz', 'jazz-rock']
    genres = set(meta['genres'])
    index = [(i, set( s.split('-'))) for i,s in enumerate(datasets) if set(s.split('-'))==genres][0][0]

    sets = {'training':'./Datasets/'+datasets[index]+'.dat',
            'control': './Datasets/'+datasets[index]+'_test.dat',
            'swapping': './Datasets/'+datasets[index]+'_swap.dat'}
    return sets[tag]



def printHistoryValues(tags):
    rows = [[h[tag] for tag in tags] for h in history]
    #todo: pretty format the rows in a kind of table



def matches(fslist):
    return [(x,y, len(set(fslist[x]) & set(fslist[y]))) for x in fslist for y in fslist if x!=y]


def showHistoryErrors():
    for h in history:
        print('\n\n----',h['training set']['path'], '(spatial:',h['spatial'],')')
        ins, outs, l, _ = datatools.preparedata(h['training set']['path'], h['spatial'])
        datatools._showerrors(h['training set']['path'], ins, outs, 'train', h['best net'])


if __name__=='__main__':

    algorithms = ['NEAT standard', 'Mutating Training Set NEAT']
    spatials = [1,2,3]
    datasets = ['./Datasets/classic-jazz-rock.dat', './Datasets/classic-rock.dat', './Datasets/classic-jazz.dat',
                './Datasets/jazz-rock.dat']

    #for d,a,s in itertools.product(datasets, algorithms, spatials):
    #    print(d,a,'(spatial:',s,') ',len([x for x in history if x['algorithm']==a and x['spatial']==s and x['training set']['path']==d]))

    """for d in [x for x in history if x['training set']['path']==datasets[1] and x['spatial']==2]:
        print('\n\n################################', d['generations'],'algorithm:', d['algorithm'])
        datatools.showErrors(d)

    exit(0)"""

    for d,a,s in itertools.product(datasets, algorithms, spatials):
        values = [x['control score'] for x in history if x['algorithm'] == a and
                                                            x['spatial']==s and datatools.compatible(datatools.metadata(d),x['training set'])]
        if len(values)>0:
            print(d,a,s,' \t',len(values),':', averagePerformance(d,a,s), sum(values)/len(values))


    for x in itertools.product(datasets, algorithms, spatials):
        plotRank(x[0],x[1],x[2])
        #showrank(x[0],x[1],x[2])

    #for d in datasets:
    #   x = bestPerformer(d)
    #   print(d,':',x[0],x[1]['algorithm'], '(spatial:',x[1]['spatial'],')', x[1]['generations'], x[1]['control score'], len(x[1]['control errors']))








