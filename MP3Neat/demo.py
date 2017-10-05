import datatools
from enums import RegistryKey

files=['./MIDI/Jazz/route_66_gw.mid', './MIDI/ClassicMusic/furelis.mid',
                                              './MIDI/Rock/1323.mid']
register = './register.dat'



def classifyMidi(*toClassify, reg = register):
    res = datatools.MIDIExtractor().classify(toClassify,
                                             orderSelectionCriterium=min,
                                             runEvaluationCriterium=lambda h: 1 - (
                                             h[RegistryKey('control errors')] / h[RegistryKey('control set')]['size']),
                                             register=reg)

    outputmessage = "-------- Classification results ----- \n\n"

    for key in res[0]:
        outputmessage+= str(key) + '\t' + str(res[0][key][0]) + ' (' + str(1-res[0][key][1]) + ')\n'

    print('\n\n\n'+outputmessage)

    return res[0], outputmessage


#classifyMidi(files[0],files[1])



