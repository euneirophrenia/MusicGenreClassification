import datatools
import matplotlib.pyplot as plot
import matplotlib.patches as patches

history = datatools.loadAllfromPickle('./register.dat')


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
    _, _, labels, _ = datatools.preparedata(dataset, order=spatial)
    allfeatuers = [labels[x] for x in labels if x < 0]
    data = [x for x in history if  x['algorithm']==algorithm and x['spatial']==spatial
            and datatools.compatible(datatools.metadata(dataset), x['training set'])
            and x['best genome'] is not None and x['generations']>100] #gen>100 to exclude some test runs to debug problems

    if len(data)==0:
        print("No run found")
        return {},0

    ranked = {key : {f :0.0 for f in allfeatuers} for key in data[0]['training set']['genres']}
    if spatial==1:
        ranked['Output'] = {f :0.0 for f in allfeatuers} ## workaround

    scoresum=0.0

    for datum in data:
        best = datum['best genome']
        scoresum+= datum['control score'] ** 2

        for i in range(0, datum['spatial']):
            crunch_output_node(i, ranked[labels[i]], best, labels)


        for k in ranked:
            for f in ranked[k]:
                ranked[k][f] *= datum['control score']**2

    for k in ranked:
        for f in ranked[k]:
            ranked[k][f] /= scoresum


    if spatial == 1:
        meta = datatools.metadata(dataset)
        mapping = datatools.generalMapping(meta['genres'], spatial)
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
    data = [x for x in history if datatools.compatible(x['training set'], m)]
    best = (1,None)
    for d in data:
        if 1.0*len(d['control errors'])/d['control set']['size'] < best[0]:
            best =(1.0*len(d['control errors'])/d['control set']['size'] ,d)
    return best

def averagePerformance(dataset, algorithm, spatial):
    m = datatools.metadata(dataset)
    data = [x for x in history if datatools.compatible(x['training set'], m) and x['algorithm']==algorithm and x['spatial']==spatial
            and x['generations']>100]
    total=0
    for x in data:
        total+= 1.0*len(x['control errors'])/x['control set']['size']

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
    #mng.resize(*mng.window.maxsize())

    if not andSaveThem:
        plot.show()
    else:
        raise NotImplementedError()
        #todo: fix, like actually wtf
        ##plot.draw()
        ##datatools.register(plot.figure(), output='./graphs.dat')