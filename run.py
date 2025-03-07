import sys
import numpy as np
import torch
import math
import pickle
import time

from nltk.translate.bleu_score import corpus_bleu

import utils
import model
from parameters import *

startToken = '<S>'
startTokenIdx = 0

endToken = '</S>'
endTokenIdx = 1

unkToken = '<UNK>'
unkTokenIdx = 2

padToken = '<PAD>'
padTokenIdx = 3

transToken = '<TRANS>'
transTokenIdx = 4

def perplexity(nmt, test, batchSize):
    testSize = len(test)
    H = 0.
    c = 0
    for b in range(0,testSize,batchSize):
        batch = test[b:min(b+batchSize, testSize)]
        l = sum(len(s)-1 for s in batch)
        c += l
        with torch.no_grad():
            H += l * nmt(batch)
    return math.exp(H/c)

if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    trainCorpus, devCorpus, word2ind, bpeCodesStr = utils.prepareData(sourceFileName, targetFileName, sourceDevFileName, targetDevFileName, startToken, endToken, unkToken, padToken, transToken)
    trainCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in trainCorpus ] # If word in dictionary, get its index, 
    devCorpus = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in devCorpus ] # otherwise get the index of the unknown token
    pickle.dump((trainCorpus, devCorpus), open(corpusFileName, 'wb'))
    pickle.dump(word2ind, open(wordsFileName, 'wb'))
    pickle.dump(bpeCodesStr, open(bpeCodesFileName, 'wb'))
    print('Data prepared.')

if len(sys.argv)>1 and (sys.argv[1] == 'train' or sys.argv[1] == 'extratrain'):
    (trainCorpus,devCorpus) = pickle.load(open(corpusFileName, 'rb'))
    word2ind = pickle.load(open(wordsFileName, 'rb'))

    nmt = model.LanguageModel(word2ind, startToken, endToken, unkToken, padToken, transToken).to(device)
    optimizer = torch.optim.Adam(nmt.parameters(), lr=learning_rate)

    if sys.argv[1] == 'extratrain':
        nmt.load(modelFileName)
        (iter,bestPerplexity,learning_rate,osd) = torch.load(modelFileName + '.optim')
        optimizer.load_state_dict(osd)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate
    else:
        bestPerplexity = math.inf
        iter = 0

    idx = np.arange(len(trainCorpus), dtype='int32')
    nmt.train()
    beginTime = time.time()
    for epoch in range(maxEpochs):
        np.random.shuffle(idx)
        words = 0
        trainTime = time.time()
        for b in range(0, len(idx), batchSize):
            iter += 1
            batch = [ trainCorpus[i] for i in idx[b:min(b+batchSize, len(idx))] ]
            
            words += sum( len(s)-1 for s in batch )
            H = nmt(batch)

            new_lr = embed_size**-0.5 * min(iter**-0.5, iter * warmup_steps**-1.5)
        
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            optimizer.zero_grad()
            H.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(nmt.parameters(), clip_grad)
      
            optimizer.step()
            if iter % log_every == 0:
                print("Iteration:",iter,"Epoch:",epoch+1,'/',maxEpochs,", Batch:",b//batchSize+1, '/', len(idx) // batchSize+1, ", loss: ",H.item(), "grad", grad_norm.item(), "lr", param_group['lr'] , "words/sec:",words / (time.time() - trainTime), "time elapsed:", (time.time() - beginTime) )
                trainTime = time.time()
                words = 0
                
            if iter % test_every == 0:
                nmt.eval()
                currentPerplexity = perplexity(nmt, devCorpus, batchSize)
                nmt.train()
                print('Current model perplexity: ',currentPerplexity)

                if currentPerplexity < bestPerplexity:
                    bestPerplexity = currentPerplexity
                    print('Saving new best model.')
                    nmt.save(modelFileName)
                    torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

    print('reached maximum number of epochs!')
    nmt.eval()
    currentPerplexity = perplexity(nmt, devCorpus, batchSize)
    print('Last model perplexity: ',currentPerplexity)
        
    if currentPerplexity < bestPerplexity:
        bestPerplexity = currentPerplexity
        print('Saving last model.')
        nmt.save(modelFileName)
        torch.save((iter,bestPerplexity,learning_rate,optimizer.state_dict()), modelFileName + '.optim')

if len(sys.argv)>3 and sys.argv[1] == 'perplexity':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    
    nmt = model.LanguageModel(word2ind, startToken, endToken, unkToken, padToken, transToken).to(device)
    nmt.load(modelFileName)

    processor = utils.makeProcessor(bpeCodesFileName, glossaries=["<S>", "<TRANS>", "</S>"])
    
    sourceTest = utils.readCorpus(sys.argv[2], processor)
    targetTest = utils.readCorpus(sys.argv[3], processor)
    test = [ [startToken] + s + [transToken] + t + [endToken] for (s,t) in zip(sourceTest,targetTest)]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt.eval()
    print('Model perplexity: ', perplexity(nmt, test, batchSize))

if len(sys.argv)>3 and sys.argv[1] == 'translate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind)
    print(torch.__version__)
    processor = utils.makeProcessor(bpeCodesFileName, glossaries=["<S>", "<TRANS>", "</S>"])

    # List of byte pair encoded sentences
    sourceTest = utils.readCorpus(sys.argv[2], processor) 
    test = [ [startToken] + s + [transToken] for s in sourceTest ]
    test = [ [word2ind.get(w,unkTokenIdx) for w in s] for s in test ]

    nmt = model.LanguageModel(word2ind, startToken, endToken, unkToken, padToken, transToken).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    file = open(sys.argv[3],'w', encoding="utf-8")	
    pb = utils.progressBar(2)
    pb.start(len(test))
    for s in test:
        r=nmt.generate(s) 
        st = r.index(transTokenIdx)
        result = [words[i] for i in r[st+1:-1]]
        result = ' '.join(result)+"\n"
        result = result.replace("@@ ", "")
        # print(result)
        file.write(result)	
        pb.tick()
    pb.stop()

if len(sys.argv)>2 and sys.argv[1] == 'generate':
    word2ind = pickle.load(open(wordsFileName, 'rb'))
    words = list(word2ind) # Map to list of words

    # Ignore special tokens
    processor = utils.makeProcessor(bpeCodesFileName, glossaries=["<S>", "<TRANS>", "</S>"]) 

    test = processor.process_line(sys.argv[2]).split()
    test = [word2ind.get(w,unkTokenIdx) for w in test]
    # List of indices of the words

    nmt = model.LanguageModel(word2ind, startToken, endToken, unkToken, padToken, transToken).to(device)
    nmt.load(modelFileName)

    nmt.eval()
    r=nmt.generate(test) # Return list of indices of the words
    result = [words[i] for i in r]
    result = ' '.join(result)+"\n"
    result = result.replace("@@ ", "")
    print(result)

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    processor = utils.makeProcessor(bpeCodesFileName, glossaries=["<S>", "<TRANS>", "</S>"])

    ref = [[s] for s in utils.readCorpus(sys.argv[2], processor)]
    hyp = utils.readCorpus(sys.argv[3], processor)

    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))
