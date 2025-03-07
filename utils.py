import io
import os
import pickle
import sys
import random
import nltk
from nltk.translate.bleu_score import corpus_bleu
from subword_nmt import learn_bpe, apply_bpe
nltk.download('punkt')

class progressBar:
    def __init__(self ,barWidth = 50):
        self.barWidth = barWidth
        self.period = None
    def start(self, count):
        self.item=0
        self.period = int(count / self.barWidth)
        sys.stdout.write("["+(" " * self.barWidth)+"]")
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.barWidth+1))
    def tick(self):
        if self.item>0 and self.item % self.period == 0:
            sys.stdout.write("-")
            sys.stdout.flush()
        self.item += 1
    def stop(self):
        sys.stdout.write("]\n")

def readCorpus(fileName, processor):

    print('Loading file:', fileName)
    return [ processor.process_line(line).split() for line in open(fileName, encoding="utf-8") ]

def getDictionary(corpus, startToken, endToken, unkToken, padToken, transToken, wordCountThreshold = 2):
    dictionary={}
    for s in corpus:
        for w in s:
            if w in dictionary: dictionary[w] += 1
            else: dictionary[w]=1

    words = [startToken, endToken, unkToken, padToken, transToken] + [w for w in sorted(dictionary) if dictionary[w] > wordCountThreshold]
    return { w:i for i,w in enumerate(words)}

def makeProcessor(bpeCodesFileName, glossaries):
    if not os.path.exists(bpeCodesFileName):
        print(f"Error: BPE codes file '{bpeCodesFileName}' does not exist.")
        return None

    with open(bpeCodesFileName, 'rb') as file:
        bpeCodes = pickle.load(file)

    bpeCodesIO = io.StringIO(bpeCodes)
    processor = apply_bpe.BPE(bpeCodesIO, glossaries=glossaries)
    return processor

def readRawLines(fileName):
    return [line.strip() for line in open(fileName, encoding="utf-8")]

def learnBPEMerges(lines, bpe_symbols=8000):
    """
    Takes a list of raw lines (strings), merges them into one text,
    learns BPE merges in memory, returns the merges as a string.
    """
    text_str = "\n".join(lines)
    text_io = io.StringIO(text_str)

    bpe_codes_io = io.StringIO()  # memory buffer for BPE merges output
    learn_bpe.learn_bpe(
        text_io,
        bpe_codes_io,
        num_symbols=bpe_symbols,
        is_dict=False
    )
    bpe_codes_str = bpe_codes_io.getvalue()
    return bpe_codes_str


def prepareData(sourceFileName, targetFileName, 
                sourceDevFileName, targetDevFileName, 
                startToken, endToken, 
                unkToken, padToken, 
                transToken):
    
    # Read the source and target files as a list of sentences
    print("Reading raw corpus...")
    sourceRaw = readRawLines(sourceFileName)
    targetRaw = readRawLines(targetFileName)

    # Combine the source and target files into one text
    print(f"Learning BPE codes ...")
    combinedRaw = sourceRaw + targetRaw
    # Extract Byte Pair Encoding (BPE) codes from the combined text
    bpeCodesStr = learnBPEMerges(combinedRaw)

    bpeCodesIO = io.StringIO(bpeCodesStr)
    processor = apply_bpe.BPE(bpeCodesIO)

    # Apply BPE to the source and target files
    print("Applying BPE to train data...")
    sourceBPE = [ processor.process_line(line) for line in sourceRaw ]
    targetBPE = [ processor.process_line(line) for line in targetRaw ]

    # Read the source and target development files as a list of sentences   
    sourceDevRaw = readRawLines(sourceDevFileName)
    targetDevRaw = readRawLines(targetDevFileName)

    # Apply BPE to the source and target development files
    print("Applying BPE to dev data...")
    sourceDevBPE = [ processor.process_line(line) for line in sourceDevRaw ]
    targetDevBPE = [ processor.process_line(line) for line in targetDevRaw ]

    # Convert BPE'd lines into lists of tokens 
    sourceCorpus = [line.split() for line in sourceBPE]
    targetCorpus = [line.split() for line in targetBPE]
    sourceDev    = [line.split() for line in sourceDevBPE]
    targetDev    = [line.split() for line in targetDevBPE]

    # Build dictionary from combined BPE tokens
    word2ind = getDictionary(sourceCorpus + targetCorpus, 
                             startToken, endToken, unkToken, 
                             padToken, transToken)

    # Insert special tokens around each line
    trainCorpus = [
        [startToken] + s + [transToken] + t + [endToken]
        for (s, t) in zip(sourceCorpus, targetCorpus)
    ]
    devCorpus = [
        [startToken] + s + [transToken] + t + [endToken]
        for (s, t) in zip(sourceDev, targetDev)
    ]

    print('Corpus loading + BPE completed.')
    return trainCorpus, devCorpus, word2ind, bpeCodesStr

