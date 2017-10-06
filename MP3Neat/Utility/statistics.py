import datatools
import matplotlib.pyplot as plot
import matplotlib.patches as patches
from enums import RegistryKey

datamanager = datatools.DataManager()
history = datamanager.get('./register.dat')



def crunch_output_node(keynode, tempvals, best, labels, weight=1, visited=[]):
    features = [(key[0], best.connections[key].weight) for key in best.connections if key[1] == keynode and key[0] < 0
                and best.connections[key].enabled]
    othernodes = [(key[0], best.connections[key].weight) for key in best.connections if key[1] == keynode and key[0] >= 0
                  and best.connections[key].enabled]


    for feat in features:
        tempvals[labels[feat[0]]]+=feat[1]*weight

    for other in othernodes:
        if other not in visited:
            visited.append(other)
            tempvals = crunch_output_node(other[0], tempvals, best, labels, weight=weight*other[1], visited=visited)

    return tempvals



def rank(dataset, algorithm, spatial):
    data = [x for x in history if  x[RegistryKey.ALGORITHM]==algorithm and x[RegistryKey.OUTPUT_DIMENSION]==spatial
            and datatools.DataManager.compatible(datamanager.metadata(dataset), x[RegistryKey.TRAIN_SET])
            and x[RegistryKey.BEST_GENOME] is not None and x[RegistryKey.GENERATIONS]>100] #gen>100 to exclude some test runs to debug problems

    if len(data)==0:
        return {},0

    _, _, labels, _ = datamanager.preparedata(dataset, order=spatial)
    allfeatuers = [labels[x] for x in labels if x < 0]

    ranked = {key : {f :0.0 for f in allfeatuers} for key in data[0][RegistryKey.TRAIN_SET]['genres']}
    if spatial==1:
        ranked['Output'] = {f :0.0 for f in allfeatuers} ## workaround

    scoresum=0.0

    for datum in data:
        best = datum[RegistryKey.BEST_GENOME]
        scoresum+= datum[RegistryKey.CONTROL_SCORE] ** 2

        for i in range(0, datum[RegistryKey.OUTPUT_DIMENSION]):
            crunch_output_node(i, ranked[labels[i]], best, labels)


        for k in ranked:
            for f in ranked[k]:
                ranked[k][f] *= datum[RegistryKey.CONTROL_SCORE]**2

    for k in ranked:
        for f in ranked[k]:
            ranked[k][f] /= scoresum


    if spatial == 1:
        meta = datamanager.metadata(dataset)
        mapping = datatools.Utility.generalMapping(meta['genres'], spatial)
        for key in ranked:
            if key != 'Output':
                for f in ranked[key]:
                    ranked[key][f] = ranked['Output'][f] * mapping[key][0] if mapping[key][0]>0 else -ranked['Output'][f]

    ranked.pop('Output',None)

    maximum = max([x for genre in ranked for x in ranked[genre].values()])
    minimum = min([x for genre in ranked for x in ranked[genre].values()])
    for genre in ranked:
        for f in ranked[genre]:
                ranked[genre][f] = (ranked[genre][f])/(maximum - minimum)

    return ranked, len(data)

def bestPerformer(dataset):
    m = datatools.metadata(dataset)
    data = [x for x in history if datatools.DataManager.compatible(x[RegistryKey.TRAIN_SET], m)]
    best = (1,None)
    for d in data:
        if 1.0*d[RegistryKey.CONTROL_ERRORS]/d[RegistryKey.CONTROL_SET]['size'] < best[0]:
            best =(1.0*d[RegistryKey.CONTROL_ERRORS]/d[RegistryKey.CONTROL_SET]['size'] ,d)
    return best

def averagePerformance(dataset, algorithm, spatial):
    m = datamanager.metadata(dataset)
    data = [x for x in history if datatools.DataManager.compatible(x[RegistryKey.TRAIN_SET], m) and x[RegistryKey.ALGORITHM]==algorithm
            and x[RegistryKey.OUTPUT_DIMENSION]==spatial
            and x[RegistryKey.GENERATIONS]>100]
    total=0
    for x in data:
        total+= 1.0*x[RegistryKey.CONTROL_ERRORS]/x[RegistryKey.CONTROL_SET]['size']

    return 1.0 - total/len(data) if len(data)>0 else 0

def showrank(dataset, algorithm, order):
    print('\n\n\n----------',dataset, algorithm, order,'-------------')
    res = rank(dataset, algorithm, order)
    print(res[1],' run found.')
    for genre in res[0]:
        print('\n-----', genre, '----')
        for key in res[0][genre]:
            if res[0][genre][key] != -0: ########################################################
                print(key, ':', res[0][genre][key])


def plotRank(dataset, algorithm, order, ranked=None, andSaveThem = False):
    if ranked is None:
        ranked = rank(dataset, algorithm, order)
    if ranked[1]==0:
        return

    ##TITOLO##
    genres = ''
    for g in sorted(ranked[0]):
        genres+= g + ' - '

    plot.title(r"$\bf{Dataset}$ " + genres[:-3] + '\n'+r'$\bf{Algotithm}$ ' + algorithm + '\n'+r"$\bf{Output dimension}$: " + str(order) # +
               #r', $\bf{runs}$: ' + str(ranked[1])
              , fontname='Dejavu Serif')

    #LEGENDA
    colors = {'classic':'blue','jazz':'red','rock':'green'}
    plot.legend(handles=[patches.Patch(color=colors[key],label=key) for key in ranked[0]])

    #Score Medio
    plot.gcf().text(0.13,0.92, "Average Score: "+str(round(100*averagePerformance(dataset,algorithm,order),4))+"%",
                    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'), weight='bold'
                    )

    #DATA
    usedfeats = {j: f for genre in ranked[0] for j, f in enumerate(sorted(ranked[0][genre])) if ranked[0][genre][f] != 0}

    xs = {genre : [j for j,f in enumerate(sorted(ranked[0][genre])) if ranked[0][genre][f]!=0] for genre in ranked[0]}
    ys = {genre: [ranked[0][genre][f] for f in ranked[0][genre] if f in usedfeats.values()] for genre in ranked[0]}

    #Compressione e normalizzazione degli intervalli
    newf = {}
    for j, f in enumerate(sorted(usedfeats)):
        newf[f]=j

    for genre in ranked[0]:
        newxs = sorted(newf.values())
        plot.plot(newxs, ys[genre], color=colors[genre])

    plot.xticks(range(0, len(newf)), [usedfeats[j] for j in newf] , rotation=90, size=7)

    #Other stuff
    plot.grid()
    plot.gcf().subplots_adjust(bottom=0.3)
    mng=plot.get_current_fig_manager()
    mng.frame.Maximize(True)

    if not andSaveThem:
        plot.show()
    else:
        raise NotImplementedError()
        #todo: fix, like actually wtf
        ##plot.draw()
        ##datatools.register(plot.figure(), output='./graphs.dat')