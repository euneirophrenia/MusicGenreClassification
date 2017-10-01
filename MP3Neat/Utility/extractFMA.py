import csv
import datatools

def findById(data, tid):
    for x in data:
        if int(x['track_id'])== tid:
            return x
    return None


data = []
datamanager=datatools.DataManager()

with open('raw_tracks.csv', encoding='utf-8') as csvfile:
    keys = csvfile.readline().replace('\n','').split(',')
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        elem = {}
        for key, value in zip(keys, row):
            elem[key]=value

        data.append(elem)


usefuldata = [x for x in data if len(x['track_genres'])>0]
feats = datamanager._loadAllFromACEXml('./fma.xml')

for f in feats:
    trackid = int(str(f['title']).split('.')[0])
    meta = findById(usefuldata, trackid)
    if meta is not None:
        for key in meta:
            f[key] = meta[key]

        datatools.DataManager._savePickle(f, output='./rawfma.dat')

