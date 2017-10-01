from multiprocessing import Pool
from os import *
from os.path import *

import music21.converter as conv
import music21.features.jSymbolic as features

#### NOT REALLY WORTH THE PAIN AND THE HUUUUUGE ASS CPU TIME ##########

####### TOTALLY OUTDATED, KEEPING JIC (just in case) ##########

alreadydone=[features.AcousticGuitarFractionFeature, features.StringKeyboardFractionFeature,
            features.ElectricInstrumentFractionFeature, features.ElectricGuitarFractionFeature,
            features.ViolinFractionFeature, features.SaxophoneFractionFeature,
            features.BrassFractionFeature, features.WoodwindsFractionFeature,
            features.OrchestralStringsFractionFeature, features.StringEnsembleFractionFeature,
            features.AmountOfArpeggiationFeature, features.AverageTimeBetweenAttacksFeature,
            features.DirectionOfMotionFeature, features.DurationOfMelodicArcsFeature,
            features.ImportanceOfBassRegisterFeature, features.ImportanceOfMiddleRegisterFeature,
            features.ImportanceOfHighRegisterFeature, features.NoteDensityFeature,
            features.PitchVarietyFeature, features.RepeatedNotesFeature,
            features.SizeOfMelodicArcsFeature, features.VariabilityOfTimeBetweenAttacksFeature]

def loadExtractors(all=False):
    fs = features.extractorsById
    if all:
        return [fs[k][i] for f in fs for k in fs for i in range(len(fs[k]))
                if fs[k][i] is not None and fs[k][i] in features.featureExtractors]

    return [fs[k][i] for f in fs for k in fs for i in range(len(fs[k]))
            if fs[k][i] is not None and fs[k][i] in features.featureExtractors and fs[k][i] not in alreadydone]

    """return [features.AcousticGuitarFractionFeature, features.StringKeyboardFractionFeature,
            features.ElectricInstrumentFractionFeature, features.ElectricGuitarFractionFeature,
            features.ViolinFractionFeature, features.SaxophoneFractionFeature,
            features.BrassFractionFeature, features.WoodwindsFractionFeature,
            features.OrchestralStringsFractionFeature, features.StringEnsembleFractionFeature,
            features.AmountOfArpeggiationFeature, features.AverageTimeBetweenAttacksFeature,
            features.DirectionOfMotionFeature, features.DurationOfMelodicArcsFeature,
            features.ImportanceOfBassRegisterFeature, features.ImportanceOfMiddleRegisterFeature,
            features.ImportanceOfHighRegisterFeature, features.NoteDensityFeature,
            features.PitchVarietyFeature, features.RepeatedNotesFeature,
            features.SizeOfMelodicArcsFeature, features.VariabilityOfTimeBetweenAttacksFeature]"""

"""    return [features.AcousticGuitarFractionFeature, features.AmountOfArpeggiationFeature,
            features.AverageMelodicIntervalFeature, features.AverageNoteDurationFeature,
            features.AverageNumberOfIndependentVoicesFeature, features.AverageTimeBetweenAttacksFeature,
            features.BrassFractionFeature, features.ChangesOfMeterFeature,
            features.ChromaticMotionFeature, features.CompoundOrSimpleMeterFeature,
            features.DirectionOfMotionFeature, features.AverageVariabilityOfTimeBetweenAttacksForEachVoiceFeature,
            features.AverageNumberOfIndependentVoicesFeature, features.DistanceBetweenMostCommonMelodicIntervalsFeature,
            features.DominantSpreadFeature, features.DurationOfMelodicArcsFeature,
            features.ElectricGuitarFractionFeature, features.ElectricInstrumentFractionFeature,
            features.ImportanceOfBassRegisterFeature, features.ImportanceOfHighRegisterFeature,
            features.ImportanceOfMiddleRegisterFeature, features.InitialTempoFeature,
            features.IntervalBetweenStrongestPitchClassesFeature, features.IntervalBetweenStrongestPitchesFeature,
            features.MaximumNoteDurationFeature, features.MaximumNumberOfIndependentVoicesFeature,
            features.MelodicFifthsFeature, features.MelodicOctavesFeature,
            features.MelodicThirdsFeature, features.MelodicTritonesFeature,
            features.MinimumNoteDurationFeature, features.MostCommonMelodicIntervalFeature,
            features.MostCommonMelodicIntervalPrevalenceFeature, features.MostCommonPitchClassFeature,
            features.MostCommonPitchClassPrevalenceFeature, features.MostCommonPitchFeature,
            features.MostCommonPitchPrevalenceFeature, features.NoteDensityFeature,
            features.NumberOfCommonMelodicIntervalsFeature, features.NumberOfCommonPitchesFeature,
            features.NumberOfPitchedInstrumentsFeature, features.OrchestralStringsFractionFeature,
            features.PitchClassVarietyFeature, features.PitchVarietyFeature,
            features.PrimaryRegisterFeature, features.QuintupleMeterFeature,
            features.RangeFeature, features.RelativeStrengthOfMostCommonIntervalsFeature,
            features.RelativeStrengthOfTopPitchClassesFeature, features.RelativeStrengthOfTopPitchesFeature,
            features.RepeatedNotesFeature, features.SaxophoneFractionFeature,
            features.SizeOfMelodicArcsFeature, features.StaccatoIncidenceFeature,
            features.StepwiseMotionFeature, features.StringEnsembleFractionFeature,
            features.StringKeyboardFractionFeature, features.StrongTonalCentresFeature,
            features.TripleMeterFeature, features.VariabilityOfNotePrevalenceOfPitchedInstrumentsFeature,
            features.VariabilityOfNumberOfIndependentVoicesFeature, features.VariabilityOfTimeBetweenAttacksFeature,
            features.ViolinFractionFeature, features.WoodwindsFractionFeature]
"""
extractors = loadExtractors()

def process(file):
    print("Extracting file ", file)
    try:
        stream=conv.parse(file)
    except Exception as e:
        print("[ERROR] Can not open ", file)
        rename(file, "./Errori/" + basename(file))
        return None
    currentpiece = {'title' : file} #todo: basename(file)
    for feature in extractors:
        try:
            currentpiece[feature.__name__] = feature(stream).extract().vector
        except Exception as e:
            print("[ERROR] ", e)
            print("          Occurred while processing ", file)
            print("          Was looking for ", feature.__name__)
            rename(file, "./Errori/" + basename(file))
            return None
    rename(file, "./Crunched/" + basename(file))
    return currentpiece

def extract(folder, procs=5, recursive=False):
    if not recursive:
        onlyfiles = [join(folder,f) for f in listdir(folder) if isfile(join(folder, f))]
    else:
        onlyfiles = [] #todo:implement lol (os.walk may do the trick)
    p = Pool(processes=procs)
    return p.imap_unordered(process, onlyfiles)

