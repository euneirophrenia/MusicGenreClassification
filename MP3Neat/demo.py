import sys

sys.path.append('./Utility')
sys.path.append('./NEAT')

import argparse
from enums import RegistryKey
import pandas

from GUI.gui import *

parser = argparse.ArgumentParser()

parser.add_argument('-r','--register',default='./saferegister.dat')
parser.add_argument('-f', '--folder', type=str)
parser.add_argument('-d','--dimension', type=str, choices=['min','max'], default='max')


def ShowGui():
    try:
        _app = wx.App(False)
        _frame = MainGUI()
        _frame.Show()
        _app.MainLoop()
    except Exception:
        pass


def classifyMidi(toClassify, reg, orderSelectionCriterium=max):
    res = datatools.MIDIExtractor().classify(toClassify,
                                             orderSelectionCriterium=orderSelectionCriterium,
                                             runEvaluationCriterium=lambda h: 1 - (
                                             h[RegistryKey('control errors')] / h[RegistryKey('control set')]['size']),
                                             register=reg)

    print( "-------- Classification results ----- \n\n")

    rows = [(str(os.path.basename(key)), str(res[0][key][0]), str(1-res[0][key][1])) for key in res[0]]

    """for key in res[0]:
        outputmessage+= str(key) + '\t' + str(res[0][key][0]) + ' (' + str(1-res[0][key][1]) + ')\n'"""

    overkill = pandas.DataFrame(rows, columns=['Title', 'Inferred genre', 'confidence'])

    print('\n\n\n')
    print(overkill)
    print('\n\n\n\n')

    return res[0], str(overkill)




args = parser.parse_args()

if args.folder is None:
    ShowGui()

else:
    files = [args.folder + f for f in os.listdir(args.folder)]
    classifyMidi(files, reg=args.register, orderSelectionCriterium=getattr(builtins,args.dimension))
