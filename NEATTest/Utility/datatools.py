import os.path as path
import os
import pickle
import xml.etree.ElementTree as etree
import numpy
import collections
import itertools
import subprocess

"""
    This module serves MANY purposes, maybe too many and I should probably split it in two.
    
    It provides functions to extract features from a midi file and to save them as well as retrieve them.
    Provides some utility functions for data aggregation and statistics, which I should probably move to another module.
    
"""


__jsymbolicjarpath = './Utility/JSymbolic/jSymbolic2.jar'

def loadAllfromPickle(filename):
    res=[]
    if not path.exists(filename):
        return res

    with open(filename, "rb") as f:
        while True:
            try:
                res.append(pickle.load(f))
            except EOFError:
                break
    return res

def loadAllFromXml(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    datas = [x for x in root if x.tag=='data_set']
    res=[]
    for datum in datas:
        current = {}
        for field in datum:
            if field.tag=='data_set_id':
                current['title']=path.basename(field.text)
            else:
                try:
                    current[field[0].text] = float(field[1].text)
                except ValueError:
                    current[field[0].text] = float(str(field[1].text).replace(',','.'))
        res.append(current)
    return res

def mergeXmlOfGenreIntoPickle(xmlfile, picklefile, genre):
    res = loadAllFromXml(xmlfile)
    with open(picklefile, "ab") as out:
        for x in res:
            x['genre'] = genre
            pickle.dump(x, out)
    print("Merged ", len(res), genre, " files.")


def generalMapping(strings, order):
    sstrings = sorted(strings)
    if order == 1:
        return {s : [v*1.0/(len(strings)-1)] for v,s in enumerate(sstrings)}

    versors = [x for x in itertools.product([0.0,1.0],repeat=order) if len([i for i in x if i==1])==1]
    mapping = {}
    for key, s in zip(versors, sstrings):
        mapping[s]=key

    if len(strings)<=order:
        return mapping

    othercombos = [x for x in itertools.product([0.0, 1.0], repeat=order) if not all(i == 0 for i in x) and x not in versors]
    remaining = sstrings[order:]

    for key, s in zip(othercombos, remaining):
        mapping[s]=key

    if len(strings)<=2**order-1:
        return mapping

    raise NotImplementedError()

def mapStringToValue(order): #original implementation, reworked in general mapping
    if order==1:
        return {'classic':0.0, 'jazz':1.0, 'rock':0.5}
    if order == 3:
        return {'classic':(0.0,0.0,1.0), 'jazz':(1.0,0.0,0.0), 'rock': (0.0,1.0,0.0)}


def compatible(m1, m2):
    return set(m1['genres']) == set(m2['genres'])


def preparedata(file, order=3, andGetHistoryBest=False):
    strings = path.basename(file).split('.')[0].split('_')[0].split('-')
    mapping = generalMapping(strings,order)
    trainset = [collections.OrderedDict(sorted(x.items())) for x in loadAllfromPickle(file)] #SORTED o le etichette vengono sbagliate!
    labels={i: x for i, x in enumerate(sorted(mapping.keys()))} if order>1 else {0:'Output'}
    cont=-1
    for key in trainset[0]: # i wish i was better at python. Future me, can you write it in a single line with list comprehension? I could not
        if type(trainset[0][key])==float:
            labels[cont]=key
            cont-=1

    hbest = None
    if andGetHistoryBest:
        history = loadAllfromPickle('./register.dat')
        meta = metadataFromList(trainset)
        historybest = [(x['best genome'], x['training score']) for x in history if compatible(x['training set'],meta)
                       and x['best genome'] is not None and x['spatial'] == order]
        hbest = [next(t)[0] for _, t in itertools.groupby(historybest, lambda x: x[1])]   # REKT GG EASY

    return [[x[key] for key in x if type(x[key])==float] for x in trainset], [mapping[x['genre']] for x in trainset], labels, hbest

def fetchmismatches(dataset, errors):
    return [loadAllfromPickle(dataset)[i] for i in errors]


def register(data, output = "./register.dat"):
    with open(output, "ab") as f:
        pickle.dump(data, f)


def historyBestWithScoreDiscrepancyBelowThreshold(threshold=0.05, filename='./register.dat'):
    return [x for x in loadAllfromPickle(filename) if abs(x['training score'] - x['control score'])<=threshold]



def mismatches(net, ins, outs, threshold=0.2, func=None):
    i = 0
    res=[]
    if func is None:
        func = lambda expected, actual: numpy.linalg.norm(numpy.array(expected) - numpy.array(actual)) >= threshold

    for xi, xo in zip(ins, outs):
        output = net.activate(xi)
        if func(xo,output):
            res.append((i,xo,output))
        i += 1
    return res


def showErrors(run, threshold=0.2):
    ins,outs,_,_ = preparedata(run['training set']['path'], order=run['spatial'])
    _showerrors(run['training set']['path'], ins, outs, 'training', run['best net'], threshold)


def _showerrors(dataset, ins, outs, tag, best, threshold=0.2):
    res = mismatches(best, ins, outs, threshold=threshold)
    misses = fetchmismatches(dataset, [x[0] for x in res])
    got = [x[2] for x in res]
    print("Errors against", tag, "data: ["+str(len(misses)) + "/" + str(len(ins))+"]")
    for i,d in enumerate(misses):
        print(d['title'], "(expected", d['genre'] + ", got ", got[i],")")
    return misses, got


def metadataFromList(lista, tag=''):
    genres = {}
    for x in lista:
        if x['genre'] in genres:
            genres[x['genre']] += 1
        else:
            genres[x['genre']] = 1
    res = dict(genres)
    res['genre count'] = len(genres)
    res['genres'] = sorted(list(genres.keys()))
    res['size'] = len(lista)
    res['path'] = rebuildPathForMeta(res, tag)
    return res

def metadata(dataset, tag=''):
    return metadataFromList(loadAllfromPickle(dataset), tag)

def concat(lista, sep='-'):
    res=''
    for x in lista:
        res+= x + sep
    return res[:-1]

def rebuildPathForMeta(meta, tag=''):
    res= './Datasets/' + concat(sorted(meta['genres']))
    if tag != '':
        res+='_' + tag
    res+='.dat'

    return res


def extractFromFolder(folderpath, kind='MIDI', outputfile='tmp.xml', definitionoutputfile='tmpdef.xml'):

    command = {'MIDI' : ['java','-jar', __jsymbolicjarpath, folderpath, outputfile, definitionoutputfile]}

    #todo: also for MP3 or other formats with other jars/utilities

    process=subprocess.Popen(command[kind])

    result = process.wait()
    if result != 0:
        raise RuntimeError("Could not properly extract the features.\nChanches are you can see the error above.")

def mergeFolderOfGenreIntoPickle(folder, picklefile, genre, kind='MIDI'):
    standardname = path.basename(folder)+'.xml'
    extractFromFolder(folder, outputfile=standardname, kind=kind, definitionoutputfile=path.basename(folder)+'def.xml')
    mergeXmlOfGenreIntoPickle(standardname, picklefile=picklefile, genre=genre)
    os.remove(standardname)
    os.remove(path.basename(folder)+'def.xml')


