import datatools

_dataset_ = './Datasets/classic-jazz-rock.dat'
_spatial_ = 1
_algorithm_ = 'NEAT standard'


_, _, labels, _ = datatools.preparedata(_dataset_, order=_spatial_)
allfeatuers = [labels[x] for x in labels if x < 0]

history = datatools.loadAllfromPickle('./register.dat')


def crunch_output_node(keynode, tempvals, best, weight=1):
    features = [(key[0], best.connections[key].weight) for key in best.connections if key[1] == keynode and key[0] < 0
                and best.connections[key].enabled]
    othernodes = [(key[0], best.connections[key].weight) for key in best.connections if key[1] == keynode and key[0] >= 0
                  and best.connections[key].enabled]


    for feat in features:
        tempvals[labels[feat[0]]]+=feat[1]*weight

    for other in othernodes:
        tempvals = crunch_output_node(other[0], tempvals, best,  weight=weight*other[1])

    return tempvals



def rank(dataset, algorithm, spatial):
    data = [x for x in history if x['algorithm']==algorithm and x['spatial']==spatial
            and datatools.compatible(datatools.metadata(dataset), x['training set'])
            and x['best genome'] is not None]

    if len(data)==0:
        print("No run found")
        return {},0



    ranked = {key : {f :0.0 for f in allfeatuers} for key in data[0]['training set']['genres']}
    ranked['Output'] = {f :0.0 for f in allfeatuers} ## workaround

    scoresum=0.0

    for datum in data:
        best = datum['best genome']
        scoresum+= datum['control score']

        for i in range(0, spatial):
            crunch_output_node(i, ranked[labels[i]], best)


        for k in ranked:
            for f in ranked[k]:
                ranked[k][f] *= datum['control score']

    for k in ranked:
        for f in ranked[k]:
            ranked[k][f] /= scoresum


    if spatial == 1:
        #todo: ammesso che i generi siano 2
        #todo: prendi ranked['Output']
        #todo: associa al genere[0] l'opposto dei valori, e al genere[1] i valori originari
        #todo: con più di 2 generi???????
        pass

    return ranked, len(data)


ranked = rank(_dataset_, _algorithm_, spatial=_spatial_)

print('Statistics made for', ranked[1], ' runs.')

for genre in ranked[0]:
    print('\n-----', genre ,'-----')
    for value in ranked[0][genre]:
        if ranked[0][genre][value]!=0:
            print(value , ' : ', ranked[0][genre][value])


### todo: controllare con il progetto main, e il relativo datatools, che dovrebbero però coincidere

#FIXED: aggiunto sorted(mapping.keys())