import os.path as path
import os, warnings
import pickle, arff
import xml.etree.ElementTree as etree
import numpy, datetime, ast
import collections, itertools
import subprocess
from enums import IODirection



_dataManagerIOFunctions = {}

def IOHandler(direction, forDatasetExtensions=[]):
    def wrapped(func):
        key = (tuple(sorted(forDatasetExtensions)),direction)
        if key in _dataManagerIOFunctions:
            raise Warning('Overriding existing ' + str(direction)+ ' function for '+ key[0])
        _dataManagerIOFunctions[key] = func
        return func
    return wrapped


class DataManager:
    """the bread and butter of everything here, from loading up datasets to saving and preparing them for the training.
    Basically, the persistance level, built to provide some caching functionalities.
    Some features are built to precisely match my needings (as the xml parsing, which works with the ACE XML format).
    To provide support for other tools a subclass / rework might be needed"""
    _datasets = {}
    _metas = {}
    __instance = None

    def __new__(cls):
        if DataManager.__instance is None:
            DataManager.__instance = object.__new__(cls)
        return DataManager.__instance

    @staticmethod
    def __properFunctionForFile(f, direction):
        among = [key[0] for key in _dataManagerIOFunctions if key[1] is direction]
        ext = str(f).split('.')[-1]
        for p in among:
            if ext in p:
                return _dataManagerIOFunctions[(p,direction)]
        raise LookupError('Unsupported dataset extension: ' + ext + '. Currently supported: '+
                          str(set(est for key in among for est in key)))

    def get(self, dataset, forceRefresh=False, andGetMeta=False, metaKeyFunction = lambda x:x['genre'], metaKeyKeyword = 'genres'):
        if dataset not in self._datasets or forceRefresh:
            self._datasets[dataset] = self.__properFunctionForFile(dataset, IODirection.Load)(dataset)
        if not andGetMeta:
            return self._datasets[dataset]

        if dataset not in self._metas or forceRefresh:
            self._metas[dataset] = DataManager.metadataFromList(self._datasets[dataset], dataset,
                                                                keyFunction=metaKeyFunction, keyKey=metaKeyKeyword)

        return self._datasets[dataset], self._metas[dataset]

    def filter(self, dataset, criterium = lambda x:True):
        return [x for x in self.get(dataset) if criterium(x)]

    def preparedata(self,file, order=3, andGetHistoryBest=False, forceRefresh=False, historypath='./register.dat'):
        rawdata, meta = self.get(file, forceRefresh=forceRefresh, andGetMeta=True)

        strings = meta['genres']
        mapping = Utility.generalMapping(strings, order)
        trainset = [collections.OrderedDict(sorted(x.items())) for x in rawdata]  # SORTED o le etichette vengono sbagliate!
        labels = {i: x for i, x in enumerate(sorted(mapping.keys()))} if order > 1 else {0: 'Output'}
        cont = -1
        for key in trainset[0]:  # i wish i was better at python. Future me, can you write it in a single line with list comprehension? I could not
            if type(trainset[0][key]) == float:
                labels[cont] = key
                cont -= 1

        hbest = None
        if andGetHistoryBest:
            history = self.get(historypath)
            hbest = [x['best genome'] for x in history if
                           DataManager.compatible(x['training set'], meta) and
                            x['best genome'] is not None and
                            x['spatial'] == order and
                            x['generations'] > 200]


        return [[x[key] for key in x if type(x[key]) == float] for x in trainset], \
               [Utility.mapListOfGenresToVector(x['genre'], mapping) for x in trainset], \
               labels, hbest

    @staticmethod
    @IOHandler(IODirection.Save, forDatasetExtensions=['dat', 'pickle'])
    def _savePickle(data, output='./register.dat'):
        with open(output, "ab") as f:
            pickle.dump(data, f)


    @staticmethod
    @IOHandler(IODirection.Save, forDatasetExtensions=['arff'])
    def _saveArff(data, output):
        attrs, raw = DataManager.toArff(data)
        converted = {
            'description':'',
            'relation':output,
            'attributes': attrs,
            'data':raw
        }
        with open(output, 'w') as file:
            arff.dump(converted, file)

    @staticmethod
    def save(data, outputfile):
        DataManager.__properFunctionForFile(outputfile, IODirection.Save)(data, outputfile)

    @staticmethod
    @IOHandler(IODirection.Load, forDatasetExtensions=['dat', 'pickle'])
    def _loadAllfromPickle(filename):
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

    @staticmethod
    @IOHandler(IODirection.Load, forDatasetExtensions=['xml'])
    def _loadAllFromACEXml(filename):
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
                    name_tag = '' if len(field)==2 else '_0'
                    for i in range(1, len(field)):
                        try:
                            current[field[0].text + name_tag] = float(field[i].text)
                        except ValueError:
                            current[field[0].text + name_tag] = float(str(field[i].text).replace(',','.'))
                        name_tag = '_'+str(i)
            res.append(current)
        return res

    @staticmethod
    def updateData(newones, oldones, outputfile, mergekey = 'title'):
        processedoldones = {y[mergekey] : y for y in oldones}
        for x in newones:
            if x[mergekey] not in processedoldones:
                DataManager.save(x, outputfile)
            else:
                corr = processedoldones[x[mergekey]]
                res = corr.copy()
                res.update(x)
                DataManager.save(res, outputfile)


    @staticmethod
    def mergeXmlOfGenreIntoPickle(xmlfile, picklefile, genre):
        res = DataManager._loadAllFromACEXml(xmlfile)
        with open(picklefile, "ab") as out:
            for x in res:
                x['genre'] = genre
                pickle.dump(x, out)
        print("Merged ", len(res), genre, " files.")

    @staticmethod
    def rebuildPathForMeta(meta, tag=''):
        res = './Datasets/' + '-'.join(sorted(meta['genres']))
        if tag != '':
            res += '_' + tag
        res += '.dat'

        return res

    @staticmethod
    def metadataFromList(lista, datapath, keyFunction = lambda x : x['genre'], keyKey = 'genres'):
        genres = {}
        for x in lista:
            for g in keyFunction(x):
                if g in genres:
                    genres[g] += 1
                else:
                    genres[g] = 1
        res = dict(genres)
        res[keyKey+' count'] = len(genres)
        res[keyKey] = sorted(list(genres.keys()))
        res['size'] = len(lista)
        res['path'] = datapath
        res['featurecount'] = len(lista[0])
        res['filetype'] = 'MP3' if 'MP3' in datapath.split('/') else 'MIDI' ##hack, gotta think about something better
        ##maybe change persistance to a format capable of holdin' metadatas OR encode the metadata as the first element
        ##1st element would require some code convention, tho
        return res

    def metadata(self,dataset, forceRefresh=False, metaKeyFunction = lambda x:x['genre'], metaKeyKeyword='genres'):
        return self.get(dataset, andGetMeta=True, forceRefresh=forceRefresh, metaKeyKeyword=metaKeyKeyword,
                        metaKeyFunction=metaKeyFunction)[1]

    @staticmethod
    def compatible(m1, m2):
        return set(m1['genres']) == set(m2['genres']) and m1['filetype'] == m2['filetype'] and\
               m1['featurecount'] == m2['featurecount']


    @staticmethod
    def toArff(data):
        py_to_arff = {float:'REAL', str:'STRING', int:'REAL', datetime.datetime:'DATE', list:'STRING', dict:'STRING', tuple:'STRING'}
        if len(data)<1:
            return [],[]
        feats = sorted(data[0].keys())
        attrs =[(f, py_to_arff[type(data[0][f])]) for f in feats]
        data = [[x[f] for f in feats] for x in data]
        return attrs, data

    @staticmethod
    def parseArff(arffobject):
        arff_to_py = {'REAL':float, 'STRING':str, 'DATE':datetime.datetime}
        labels = [x[0] for x in arffobject['attributes']]
        toactivate = [x[0] for x in arffobject['attributes'] if x[1]=='STRING']
        res=[]
        for datum in arffobject['data']:
            current={}
            for i,feature in enumerate(datum):
                current[labels[i]] = arff_to_py[arffobject['attributes'][i][1]](feature) if feature is not None else None
                if feature in toactivate:
                    current[labels[i]] = ast.literal_eval(str(current[labels[i]]).replace('\\',''))
            res.append(current)
        return res

    @staticmethod
    #@loadingFunction(forDatasetExtensions=['arff'])
    @IOHandler(IODirection.Load, forDatasetExtensions=['arff'])
    def _loadAllFromArff(file):
        with open(file, 'r') as f:
            res=arff.load(f)
        return DataManager.parseArff(res)


class ErrorFetcher:
    """Just convenience methods to fetch precise records from dataset"""
    __instance = None
    __datamanager = DataManager()
    def __new__(cls):
        if ErrorFetcher.__instance is None:
            ErrorFetcher.__instance = object.__new__(cls)
        return ErrorFetcher.__instance

    def fetchmismatches(self,dataset, errors):
        data = self.__datamanager.get(dataset)
        return [data[i] for i in errors]


    def historyBestWithScoreDiscrepancyBelowThreshold(self,threshold=0.05, filename='./register.dat'):
        return [x for x in self.__datamanager.get(filename) if abs(x['training score'] - x['control score'])<=threshold]


    @staticmethod
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


    def showErrors(self,run, threshold=0.2):
        ins,outs,_,_ = self.__datamanager.preparedata(run['training set']['path'], order=run['spatial'])
        self._showerrors(run['training set']['path'], ins, outs, 'training', run['best net'], threshold)


    def _showerrors(self,dataset, ins, outs, tag, best, threshold=0.2):
        res = ErrorFetcher.mismatches(best, ins, outs, threshold=threshold)
        misses = self.fetchmismatches(dataset, [x[0] for x in res])
        got = [x[2] for x in res]
        print("Errors against", tag, "data: ["+str(len(misses)) + "/" + str(len(ins))+"]")
        for i,d in enumerate(misses):
            print(d['title'], "(expected", str(d['genre']) + ", got ", got[i],")")
        return misses, got


##TODO:: Right now it works under the assumption that the extractor is a jar file
##todo:: generalize to make them accept a custom command to run, maybe defaulting to a jar launch or something
class Extractor:
    """Base class for automatic feature extraction and dataset creation.
    While this is abstract, you can add several subclasses to provide support for different file formats.
    Right now, jSymbolic and jAudio are used to extract from MIDI and WaveForm formats.
    TODO: Rework to not necessarly depend on external resources, mostly removing the "path" dependency."""

    def __init__(self):
        raise NotImplementedError('this is supposed to be an abstract class man')

    def __clean(self, filename, folder):
        pass

    def extractFromFolder(self,folder, outputfile='temp.xml', definitionoutputfile='tempdef.xml'):
        pass

    def extractFromList(self, lista, outputfile='temp.xml', definitionoutputfile='tempdef.xml'):
        pass


    def _prepareFiles(self, files, orderSelectionCriterium, amongGenres, netSelectionCriterium, register):
        actual = [x for x in files if str(x).split('.')[-1] in self.managed_extensions]
        data = [collections.OrderedDict(sorted(x.items())) for x in self.extractFromList(actual)]

        aggregatedHistory = Utility.historyBestByOrderAndGenre(netSelectionCriterium, register=register, sep='#')
        if len(aggregatedHistory) == 0:
            print('No history found, check paths. Or possibly not a single run was found')
            return

        if amongGenres is None:
            genres = max(aggregatedHistory.keys(), key=lambda s: len(s[1].split('#')))[1]
        else:
            genres = '#'.join(sorted(list(map(lambda x: x.lower(), amongGenres))))

        bestorder = orderSelectionCriterium(list(map(lambda x: x[0], [key for key in aggregatedHistory if key[1] == genres])))
        bestnet, controlscore = aggregatedHistory[(bestorder, genres)]

        return data, bestnet, bestorder, sorted(genres.split('#')), controlscore

    def classify(self, files, orderSelectionCriterium = max, amongGenres = None,
                 runEvaluationCriterium = lambda h : 1 - (len(h['control errors']) / h['control set']['size']),
                 register = './register.dat'):

        data, bestnet, bestorder, genres, controlscore = self._prepareFiles(files, orderSelectionCriterium, amongGenres,
                                                                            runEvaluationCriterium, register)

        inverseMap = Utility.inverseMapping(genres, bestorder)
        mapkeys = list(inverseMap.keys())
        res = {}

        for datum in data:
            net_input = [datum[key] for key in sorted(list(datum.keys())) if type(datum[key]) == float]
            output = Utility.closestVector(bestnet.activate(net_input) ,mapkeys)
            res[datum['title']] = inverseMap[output[0]], output[1]

        return res, genres, bestorder, controlscore

    def mergeFolderOfGenreIntoPickle(self,folder, picklefile, genre):
        standardname = path.basename(folder) + '.xml'
        outputfile = self.extractFromFolder(folder, outputfile=standardname,
                          definitionoutputfile=path.basename(folder) + 'def.xml')
        DataManager.mergeXmlOfGenreIntoPickle(outputfile, picklefile=os.path.abspath(picklefile), genre=genre)
        self.__clean(outputfile, folder)

class MIDIExtractor(Extractor):
    managed_extensions = ['mid','midi','mei']
    def __init__(self, pathtolib='./Utility/JSymbolic/jSymbolic2.jar'):
        self.path = pathtolib

    def extractFromFolder(self,folder, outputfile='temp.xml', definitionoutputfile='tempdef.xml'):
        command = ['java','-jar', self.path, folder, outputfile, definitionoutputfile]
        process=subprocess.Popen(command)
        result = process.wait()
        if result != 0:
            raise RuntimeError("Could not properly extract the features. Chanches are you can see the error above.")
        return outputfile

    def extractFromList(self, lista, outputfile='temp.xml', definitionoutputfile='tempdef.xml'):
        res = []
        for file in lista:
            temp=self.extractFromFolder(file, outputfile, definitionoutputfile)
            res.extend(DataManager._loadAllFromACEXml(temp))
        self.__clean(outputfile, definitionoutputfile)
        return res

    def __clean(self, standardname, definition):
        try:
            os.remove(standardname)
            os.remove(path.basename(definition))
        except Exception as e:
            print('Problems cleaning up.' + str(e))


class WaveFormExtractor(Extractor):
    managed_extensions = ['mp3','wav','ogg', 'wave','aif','aiff','aifc','au','snd','oga']
    def __init__(self, pathtoworkingdir = './Utility/JAudio/', pathtolib='./jAudio.jar'):
        self.path = pathtolib
        self.pathtoworkingdir = pathtoworkingdir
        self.__curdir = os.getcwd()

    def extractFromFolder(self,folder, outputfile='temp', definitionoutputfile='tempdef.xml'):
        folder = path.abspath(folder)
        if not str(folder).endswith('/'):
            folder = folder + '/'
        files = [folder + file for file in os.listdir(folder) if file.split('.')[-1] in self.managed_extensions]
        return self.extractFromList(files, outputfile, definitionoutputfile)

    def __clean(self, filename, folder):
        try:
            os.remove(filename)
            os.remove(path.basename(folder) + '.xmlFK.xml')
            os.chdir(self.__curdir)
        except Exception as e:
            print('Problems cleaning up.' + str(e))

    def extractFromList(self, lista, outputfile='temp.xml', definitionoutputfile='tempdef.xml'):
        command = ['java', '-jar', self.path, '-s', './settings.xml', outputfile]
        for f in lista:
            command.append(f)

        os.chdir(self.pathtoworkingdir)
        process = subprocess.Popen(command)
        result = process.wait()
        if result != 0:
            raise RuntimeError("Could not properly extract the features. Chanches are you can see the error above.")
        res = DataManager._loadAllFromACEXml(outputfile + 'FV.xml')
        self.__clean(outputfile + 'FV.xml', definitionoutputfile)
        return res

    def classify(self, files, orderSelectionCriterium = max, amongGenres = None,
                 runEvaluationCriterium = lambda h : 1 - (len(h['control errors']) / h['control set']['size']),
                 register = './register.dat', threshold=0.2):

        data, bestnet, bestorder, genres, controlscore = Extractor._prepareFiles(self, files, orderSelectionCriterium,
                                                                                 amongGenres,
                                                                                 runEvaluationCriterium, register)

        res = {}

        for datum in data:
            net_input = [datum[key] for key in sorted(list(datum.keys())) if type(datum[key]) == float]
            output = bestnet.activate(net_input)
            res[datum['title']] = Utility.mapBackVectorToVersorGenres(output, genres, threshold=threshold)

        return res, genres, bestorder, controlscore



class Utility:
    """Collection of utility methods that were useful at some point in time."""

    @staticmethod
    def _correct(basename, parent):
        tokens = basename.split('.')
        tokens.insert(len(tokens) - 1, os.path.basename(parent) + ".")
        return ''.join(tokens)

    @staticmethod
    def flatten(basefolder, andClean=False):
        """Flattens a file system, moving all the files in subfolders to top level. Because of reasons."""
        for f in os.listdir(basefolder):
            current = os.path.join(basefolder, f)
            if os.path.isdir(current):
                for x in os.listdir(current):
                    if os.path.isfile(os.path.join(current, x)):
                        try:
                            os.rename(os.path.join(current, x), os.path.join(basefolder, os.path.basename(x)))
                        except FileExistsError:
                            os.rename(os.path.join(current, x),
                                      os.path.join(basefolder, Utility._correct(os.path.basename(x), current)))
                    if os.path.isdir(os.path.join(current, x)):
                        Utility.flatten(current, andClean=andClean)
        if andClean:
            Utility.clean(basefolder)

    @staticmethod
    def clean(basefolder):
        """Cleans empty subfolders"""
        for f in os.listdir(basefolder):
            current = os.path.join(basefolder, f)
            if os.path.isdir(current) and len(os.listdir(current)) == 0:
                os.rmdir(current)

    @staticmethod
    def aggregate(lista):
        res = {}
        for x in lista:
            if x in res:
                res[x] += 1
            else:
                res[x] = 1
        return res

    @staticmethod
    def historyBestByOrderAndGenre(runEvaluationCriterium, sep='#', register ='./register.dat'):
        history = [x for x in DataManager().get(register) if x['generations']>500 and x['best net'] is not None]
        aggregated = {}
        for h in history:
            key = sep.join(sorted(list(map(lambda x: x.lower(), h['training set']['genres'])))) # hacks
            if (h['spatial'], key) not in aggregated:
                bestrun = max([x for x in history if h['spatial']==x['spatial']
                                                        and key == sep.join(sorted(list(map(lambda x: x.lower(),
                                                                                       x['training set']['genres']))))],
                              key=runEvaluationCriterium)

                aggregated[(h['spatial'], key)] = bestrun['best net'], runEvaluationCriterium(bestrun)

        return aggregated


    @staticmethod
    def generalMapping(strings, order):  ##todo::: fix this fucking shit already
        sstrings = sorted(strings)
        if order == 1:
            return {s: [v * 1.0 / (len(strings) - 1)] for v, s in enumerate(sstrings)}

        versors = numpy.identity(order)

        mapping = {}
        for key, s in zip(versors, sstrings):
            mapping[s] = key

        if len(strings) <= order:
            return mapping

        othercombos = [x for x in itertools.product([0.0, 1.0], repeat=order) if
                       not all(i == 0 for i in x) and x not in versors] ##WARNING: MAY EXPLODE (generates 2^order - order items)
        remaining = sstrings[order:]

        for key, s in zip(othercombos, remaining):
            mapping[s] = key

        if len(strings) <= 2 ** order - 1:
            return mapping

        raise NotImplementedError("I don't know if it would even make sense at this point.")

    @staticmethod
    def __mapStringToValue(order):  # original implementation, reworked in general mapping
        warnings.warn("deprecated! Don't use this.", DeprecationWarning)
        if order == 1:
            return {'classic': 0.0, 'jazz': 1.0, 'rock': 0.5}
        if order == 3:
            return {'classic': (0.0, 0.0, 1.0), 'jazz': (1.0, 0.0, 0.0), 'rock': (0.0, 1.0, 0.0)}

    @staticmethod
    def mapListOfGenresToVector(lista, mapping):
        res = numpy.zeros_like(mapping[lista[0]])
        for g in lista:
            res += numpy.array(mapping[g])
        return res #/ numpy.linalg.norm(res)   #hypersphere, remove the norm to map to a hypercube. I'd just stick to the sphere and see what happens
##hypersphere gotta be better, but for now it won't matter since only 1 genre per time is considered

    @staticmethod
    def inverseMapping(strings, order):
        direct = Utility.generalMapping(strings, order)
        return {tuple(direct[key]): key for key in direct}

    @staticmethod
    def mapBackVectorToVersorGenres(vector, genres, threshold=0.2):
        res={}
        for i,component in enumerate(vector):
            if component > threshold:
                res[genres[i]]=component
        return res

    @staticmethod
    def closestVector(v, lista):
        arrayv = numpy.array(v)
        return min([(x, numpy.linalg.norm(arrayv - numpy.array(x))) for x in lista], key= lambda couple:couple[1])


