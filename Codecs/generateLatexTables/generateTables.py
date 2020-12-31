#!/usr/bin/python
import collections
from collections import defaultdict
from itertools import islice
from ast import literal_eval
import glob, os

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

specifiedOrder = {"Rice": 0, "OptRice": 1, # bit-oriented
                  "varint-G4B": 2, "varint-G8B": 3, "varint-G8IU": 4, "varint-G8CU": 5, # byte-aligned
                  "Simple-9": 6, "Simple-16": 7, # word-aligned
                  "NewPFor": 8, "OptPFor": 9} # frame-based

def getKey(codec):
    words = codec.split("_")
    return specifiedOrder[words[0]]

def generateEfficiencyTables(outfile, misDict, bestmisDict, postpSpeedDict = None):
    datasets = set()
    
    # Convert dictionary to dictionary of lists.
    misDictList = defaultdict(list)
    misDict = collections.OrderedDict(sorted(misDict.items(), key = lambda t: t[0][2], reverse = True)) # Scalar, SSE, AVX
    misDict = collections.OrderedDict(sorted(misDict.items(), key = lambda t: t[0][1])) # Alphabetical order of datasets.
    for (codec, dataset, instructionSet), mis in misDict.items():
        datasets.add(dataset)
        misDictList[codec].append((dataset, instructionSet, mis))
    misDictList = collections.OrderedDict(sorted(misDictList.items())) # Alphabetical order of codecs.
    misDictList = collections.OrderedDict(sorted(misDictList.items(), key = lambda t: getKey(t[0]))) # Specified order of codecs.

    avgPostpSpeed = None
    if not postpSpeedDict:
        outfile.write("decoding speed. (10^6 ints/sec)\n")
    else:
        outfile.write("decompression speed. (10^6 ints/sec)\n")
        avgPostpSpeed = sum(postpSpeedDict.values()) / len(postpSpeedDict)
        outfile.write("RegularDeltaSSE speed = {0}\n".format(int(round(avgPostpSpeed))))
    outfile.write("\t" * 6)
    outfile.write(("\t" * 5).join(sorted(datasets)))
    outfile.write("\n")
    for codec, misList in misDictList.items():
        outfile.write("{0:<11}".format(codec))
        for dataset, instructionSet, mis in misList:
            i = int(round(mis if not postpSpeedDict else mis*avgPostpSpeed / (mis+avgPostpSpeed)))
            if mis == bestmisDict[dataset]:
                outfile.write(" &\\textbf{{{0:>4}}}".format(i))
            elif instructionSet == 'AVX':
                outfile.write(" &{0:>4}{1}".format(i, ' ' * 9))
            else:
                outfile.write(" &{0:>4}".format(i))
        outfile.write(" \\\\\\vspacer")
        outfile.write("\n")
    outfile.write("\n")

def generateEffectivenessTables(outfile, bpiDict, bestbpiDict):
    datasets = set()

    # Convert dictionary to dictionary of lists.
    bpiDictList = defaultdict(list)
    bpiDict = collections.OrderedDict(sorted(bpiDict.items(), key = lambda t: t[0][1])) # Alphabetical order of dataset
    for (codec, dataset), bpi in bpiDict.items():
        datasets.add(dataset)
        bpiDictList[codec].append((dataset, bpi))
    bpiDictList = collections.OrderedDict(sorted(bpiDictList.items(), key = lambda t: getKey(t[0])))

    outfile.write("bits/int.\n")
    outfile.write("\t" * 6)
    outfile.write(("\t" * 6).join(sorted(datasets)))
    outfile.write("\n")
    for codec, bpiList in bpiDictList.items():
        outfile.write("{0:<11}".format(codec))
        for dataset, bpi in bpiList:
            if bpi == bestbpiDict[dataset]:
                outfile.write("  & \\textbf{{{0:>4.2f}}}".format(round(bpi, 2)))
                outfile.write(" & " + " " * 7)
            else:
                outfile.write("  & {0:>5.2f}{1}".format(round(bpi, 2), ' ' * 8))
                loss = 100 * (round(bpi,2) - round(bestbpiDict[dataset],2)) / round(bestbpiDict[dataset],2)
                outfile.write(" & {0:>5.1f}\%".format(round(loss, 1)))
        outfile.write(" \\\\\\tspacer")
        outfile.write("\n")
    outfile.write("\n")

def next_n_lines(f, n):
    return [x.strip() for x in islice(f, n) if x.strip()]

def generateTables(datasets):
    postpSpeedVividict = Vividict()
    bpiVividict = Vividict()
    bestbpiVividict = Vividict()
    misVividict = Vividict()
    bestmisVividict = Vividict()
    for dataset in datasets:
        filename = "../stats/" + dataset + "_Delta_CPUCodecsStats.txt"
        with open(filename, 'r') as infile:
            while True:
                next_44_lines = next_n_lines(infile, 44)
                if not next_44_lines: # EOF
                    break
                
                key = None
                for i, line in enumerate(next_44_lines):
                    d = literal_eval(line)
                    if i == 0:
                        key = (d["preprocessor"], d["minlen"])
                        postpSpeedVividict[key][dataset] = d["postpSpeed"]
                        bestbpiVividict[key][dataset] = float('inf')
                        bestmisVividict[key][dataset] = 0.0
                    elif i > 1:
                        codec = d["codec"]

                        bpi = d["bits/int"]
                        bpiCodec = codec.split('_')[0]
                        bpiVividict[key][(bpiCodec, dataset)] = bpi
                        if bpi < bestbpiVividict[key][dataset]:
                            bestbpiVividict[key][dataset] = bpi

                        mis = d["decodingSpeed"]
                        misCodec = '_'.join(codec.split('_')[:-1])
                        instructionSet = codec.split('_')[-1]
                        misVividict[key][(misCodec, dataset, instructionSet)] = mis
                        if mis > bestmisVividict[key][dataset]:
                            bestmisVividict[key][dataset] = mis
        infile.closed

    for key, postpSpeedDict in postpSpeedVividict.items():
        filename = "LatexTables/CPUCodecStats_" + key[0] + "_minlen" + str(key[1]) + ".txt"
        with open(filename, 'a') as outfile:
            generateEffectivenessTables(outfile, bpiVividict[key], bestbpiVividict[key])
            generateEfficiencyTables(outfile, misVividict[key], bestmisVividict[key])
            generateEfficiencyTables(outfile, misVividict[key], bestmisVividict[key], postpSpeedDict)
        outfile.closed

def main():
    filelist = glob.glob("LatexTables/*.txt")
    for f in filelist:
        os.remove(f)

    gov2 = ['gov2', 'gov2ibda', 'gov2r', 'gov2tmf', 'gov2url']
    generateTables(gov2)
    bd = ['bd', 'bdibda', 'bdr', 'bdtmf']
    generateTables(bd)

if __name__ == "__main__":
	main()
