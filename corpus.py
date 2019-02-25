"""
This class is a modification from the publicly available repository at:
https://github.com/hazemalsaied/ATILF-LLF.v2/blob/master/Src/corpus.py
"""
import os
import operator
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

class Corpus:
    """
        a class used to encapsulate all the information of the corpus
    """

    def __init__(self, path):
        """
            an initializer of the corpus, responsible of creating a structure of objects encapsulating all the information
            of the corpus, its sentences, tokens and MWEs.

            This function iterate over the lines of corpus document to create the precedent ontology
        :param path: the physical path of the corpus document
        """
        self.sentNum, self.mweNum, self.intereavingNum, self.emeddedNum, self.singleWordExp, self.continousExp = 0, 0, 0, 0,0,0

        cuptFile = None
        if os.path.isfile(os.path.join(path, 'train.cupt')):
            cuptFile = os.path.join(path, 'train.cupt')
        #mweFile = os.path.join(path, 'train.parsemetsv')
        self.sentences = []
        devCupt = None
        if os.path.isfile(os.path.join(path, 'dev.cupt')):
            devCupt = os.path.join(path, 'dev.cupt')
        testCupt = None
        if os.path.isfile(os.path.join(path, 'test.cupt')):
            testCupt = os.path.join(path, 'test.cupt')

        blindTestCupt = None
        testBlind = os.path.join(path, 'test.blind.cupt')
        if os.path.isfile(os.path.join(path, 'test.blind.cupt')):
            blindTestCupt = os.path.join(path, 'test.blind.cupt')

        if cuptFile is not None: # and testCupt is not None:
            self.sentences, self.mweNum = Corpus.readCuptFile(cuptFile)
            #self.mweNum = Corpus.readMweFile(mweFile, self.sentences)
            self.sentNum = len(self.sentences)

            for sent in self.sentences:
                self.emeddedNum += sent.recognizeEmbededVMWEs()
                self.intereavingNum += sent.recognizeInterleavingVMWEs()
                x,y = sent.recognizeContinouosandSingleVMWEs()
                self.singleWordExp += x
                self.continousExp += y

        if devCupt is not None:    # added by me
            self.devSents, self.devMweNum = Corpus.readCuptFile(devCupt)
        if testCupt is not None:    # added by me
            self.testSents, self.testMweNum = Corpus.readCuptFile(testCupt)
        if blindTestCupt is not None:    # added by me
            self.blindTestSents = Corpus.readBlindTestFile(blindTestCupt)
        '''
        else:       # Am I right to thing that this is for those datasets that have no conllu info?
            self.sentences, self.sentNum, self.mweNum = Corpus.readSentences(mweFile)
            self.testSents, x, y = Corpus.readSentences(testBlind, forTest=True)
            for sent in self.sentences:
                self.emeddedNum += sent.recognizeEmbededVMWEs()
                self.intereavingNum += sent.recognizeInterleavingVMWEs()
                x, y = sent.recognizeContinouosandSingleVMWEs()
                self.singleWordExp += x
                self.continousExp += y
        '''

    @staticmethod
    def readCuptFile(cuptFile):
        sentences = []
        mweNum = 0      # ADDED by me
        with open(cuptFile) as corpusFile:
            # Read the corpus file
            lines = corpusFile.readlines()
            sent = None
            senIdx = 0
            sentId = ''
            sentenceText = ''
            for line in lines:
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('# sent_id:'):
                    sentId = line.split('# sentid:')[1].strip()

                elif line.startswith('# text ='):
                    sentenceText = line.split('# text =')[1].strip()    # ADDED by me
                    continue
                elif line.startswith('#'):
                    continue

                elif line.startswith('1\t'):
                    if sentId.strip() != '':
                        sent = Sentence(senIdx, sentid=sentId)
                    else:
                        sent = Sentence(senIdx)
                    senIdx += 1
                    sentences.append(sent)

                if not line.startswith('#'):
                    lineParts = line.split('\t')

                    if len(lineParts) != 11 or '-' in lineParts[0]:     # I CHANGED it to 11
                        continue
                    morpho = ''
                    if lineParts[5] != '_':
                        morpho = lineParts[5].split('|')
                    if lineParts[6] != '_' and lineParts[6] != '-':     #  I added the and part
                        token = Token(lineParts[0], lineParts[1], lemma=lineParts[2],   ## I removed .lower() from lineParts[1]
                                      abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                      dependencyParent=int(lineParts[6]),
                                      dependencyLabel=lineParts[7])
                    else:
                        print("No dependency!: ", lineParts)
                        # ME: It does never come to this else. Once it got here and it gave errors.
                        # In the case of English, this happened when the token id is 8.1, 22.1, etc.
                        token = Token(lineParts[0], lineParts[1], lemma=lineParts[2],   ## I removed .lower() from lineParts[1]
                                      abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                      dependencyLabel=lineParts[7])
                    if lineParts[4] != '_':
                        token.posTag = lineParts[4]
                    else:
                        token.posTag = lineParts[3]

                    #NEW  added by me
                    if lineParts[-1] != '*':
                        vMWEids = lineParts[-1].split(';')
                        for vMWEid in vMWEids:
                            id = int(vMWEid.split(':')[0])
                            # New MWE captured
                            if id not in sent.getWMWEIds():
                                if len(vMWEid.split(':')) > 1:
                                    type = str(vMWEid.split(':')[1])
                                    vMWE = VMWE(id, token, type)
                                else:
                                    vMWE = VMWE(id, token)
                                mweNum += 1
                                sent.vMWEs.append(vMWE)
                            # Another token of an under-processing MWE
                            else:
                                vMWE = sent.getVMWE(id)
                                if vMWE is not None:
                                    vMWE.addToken(token)
                            # associate the token with the MWE
                            token.setParent(vMWE)
                    ########################################

                        
                    # Associate the token with the sentence
                    sent.tokens.append(token)
                    sent.text += token.text + ' '
        return sentences, mweNum        #  ADDED by me

    @staticmethod
    def readBlindTestFile(cuptFile):
        sentences = []
        mweNum = 0      # ADDED by me
        with open(cuptFile) as corpusFile:
            # Read the corpus file
            lines = corpusFile.readlines()
            sent = None
            senIdx = 0
            sentId = ''
            sentenceText = ''
            for line in lines:
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('# sent_id:'):
                    sentId = line.split('# sentid:')[1].strip()

                elif line.startswith('# text ='):
                    sentenceText = line.split('# text =')[1].strip()    # ADDED by me
                    continue
                elif line.startswith('#'):
                    continue

                elif line.startswith('1\t'):
                    if sentId.strip() != '':
                        sent = Sentence(senIdx, sentid=sentId)
                    else:
                        sent = Sentence(senIdx)
                    senIdx += 1
                    sentences.append(sent)

                if not line.startswith('#'):
                    lineParts = line.split('\t')

                    if len(lineParts) != 11 or '-' in lineParts[0]:     # I CHANGED it to 11 since we have 11 columns
                        continue
                    morpho = ''
                    if lineParts[5] != '_':
                        morpho = lineParts[5].split('|')
                    if lineParts[6] != '_' and lineParts[6] != '-':     # I added the and part
                        token = Token(lineParts[0], lineParts[1], lemma=lineParts[2],   ## I removed .lower() from lineParts[1]
                                      abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                      dependencyParent=int(lineParts[6]),
                                      dependencyLabel=lineParts[7])
                    else:
                        print(lineParts)
                        # It does never come to this else. Once it got here and it gave errors.
                        token = Token(lineParts[0], lineParts[1], lemma=lineParts[2],   ## I removed .lower() from lineParts[1]
                                      abstractPosTag=lineParts[3], morphologicalInfo=morpho,
                                      dependencyLabel=lineParts[7])
                    if lineParts[4] != '_':
                        token.posTag = lineParts[4]
                    else:
                        token.posTag = lineParts[3]

                    ########################################
                       
                    # Associate the token with the sentence
                    sent.tokens.append(token)
                    sent.text += token.text + ' '
        return sentences       #  ADDED by me
    
    '''
    @staticmethod
    def readMweFile(mweFile, sentences):
        mweNum = 0
        with open(mweFile) as corpusFile:

            # Read the corpus file
            lines = corpusFile.readlines()
            noSentToAssign = False
            sentIdx = 0
            for line in lines:
                if line == '\n' or line.startswith('# sentence-text:') or (
                             line.startswith('# sentid:') and noSentToAssign) :
                    continue
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('1\t'):
                    sent = sentences[sentIdx]
                    sentIdx += 1
                lineParts = line.split('\t')
                if '-' in lineParts[0]:
                    continue
                if lineParts is not None and len(lineParts) == 4 and lineParts[3] != '_':

                    token = sent.tokens[int(lineParts[0]) - 1]
                    vMWEids = lineParts[3].split(';')
                    for vMWEid in vMWEids:
                        id = int(vMWEid.split(':')[0])
                        # New MWE captured
                        if id not in sent.getWMWEIds():
                            if len(vMWEid.split(':')) > 1:
                                type = str(vMWEid.split(':')[1])
                                vMWE = VMWE(id, token, type)
                            else:
                                vMWE = VMWE(id, token)
                            mweNum += 1
                            sent.vMWEs.append(vMWE)
                        # Another token of an under-processing MWE
                        else:
                            vMWE = sent.getVMWE(id)
                            if vMWE is not None:
                                vMWE.addToken(token)
                        # associate the token with the MWE
                        token.setParent(vMWE)
        return mweNum
    '''
    @staticmethod
    def readSentences(mweFile, forTest=False):
        sentences = []
        sentNum, mweNum = 0, 0
        with open(mweFile) as corpusFile:
            # Read the corpus file
            lines = corpusFile.readlines()
            sent = None
            senIdx = 1
            for line in lines:
                if len(line) > 0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith('1\t'):
                    # sentId = line.split('# sentid:')[1]
                    if sent is not None:
                        # Represent the sentence as a sequece of tokens and POS tags
                        sent.setTextandPOS()
                        if not forTest:
                            sent.recognizeEmbededVMWEs()
                            sent.recognizeInterleavingVMWEs()

                    sent = Sentence(senIdx)
                    senIdx += 1
                    sentences.append(sent)

                elif line.startswith('# sentence-text:'):
                    sentText = ''
                    if len(line.split(':')) > 1:
                        sent.text = line.split('# sentence-text:')[1]

                lineParts = line.split('\t')

                # Empty line or lines of the form: "8-9	can't	_	_"
                if len(lineParts) != 4 or '-' in lineParts[0]:
                    continue
                token = Token(lineParts[0], lineParts[1])
                # Trait the MWE
                if not forTest and lineParts[3] != '_':
                    vMWEids = lineParts[3].split(';')
                    for vMWEid in vMWEids:
                        id = int(vMWEid.split(':')[0])
                        # New MWE captured
                        if id not in sent.getWMWEIds():
                            type = str(vMWEid.split(':')[1])
                            vMWE = VMWE(id, token, type)
                            mweNum += 1
                            sent.vMWEs.append(vMWE)
                        # Another token of an under-processing MWE
                        else:
                            vMWE = sent.getVMWE(id)
                            if vMWE is not None:
                                vMWE.addToken(token)
                        # associate the token with the MWE
                        token.setParent(vMWE)
                # Associate the token with the sentence
                sent.tokens.append(token)
            sentNum = len(sentences)
            return sentences, sentNum, mweNum


class Sentence:
    """
       a class used to encapsulate all the information of a sentence
    """

    def __init__(self, id, sentid=''):

        self.sentid = sentid
        self.id = id
        self.tokens = []
        self.vMWEs = []
        self.identifiedVMWEs = []
        self.text = ''
        self.initialTransition = None
        self.featuresInfo = []

    @staticmethod
    def fromTextToSent(text):

        tokenizer = WordPunctTokenizer()
        wordNetLemmatiser = WordNetLemmatizer()
        sent = Sentence(0)
        sent.text = text
        tokenList = tokenizer.tokenize(text)
        posTags = nltk.pos_tag(tokenList)
        for token in tokenList:
            tokenLemma = wordNetLemmatiser.lemmatize(token)
            tokenPos = posTags[tokenList.index(token)][1]
            tokenObj = Token(tokenList.index(token), token, lemma=tokenLemma, posTag=tokenPos)
            sent.tokens.append(tokenObj)
        return sent

    def getWMWEs(self):
        return self.vMWEs

    def getWMWEIds(self):
        result = []
        for vMWE in self.vMWEs:
            result.append(vMWE.getId())
        return result

    def getVMWE(self, id):

        for vMWE in self.vMWEs:
            if vMWE.getId() == int(id):
                return vMWE
        return None

    def setTextandPOS(self):

        tokensTextList = []
        for token in self.tokens:
            self.text += token.text + ' '
            tokensTextList.append(token.text)
        self.text = self.text.strip()

    def recognizeEmbededVMWEs(self):
        if len(self.vMWEs) <= 1:
            return 0
        result = 0
        traitedPairs = []
        for vMwe1 in self.vMWEs:
            for vMwe2 in self.vMWEs:
                if vMwe1 is not vMwe2:
                    isTraited = False
                    for pair in traitedPairs:
                        if vMwe1 in pair and vMwe2 in pair:
                            isTraited = True
                    if not isTraited:
                        traitedPairs.append([vMwe1, vMwe2])
                        # Get The longer VMWE
                        masterVmwe = vMwe1
                        slaveVmwe = vMwe2
                        if len(vMwe2.tokens) > len(vMwe2.tokens):
                            masterVmwe = vMwe2
                            slaveVmwe = vMwe1
                        slaveVmwe.isEmbeded = True
                        for token in slaveVmwe.tokens:
                            if masterVmwe not in token.parentMWEs:
                                slaveVmwe.isEmbeded = False
                        if slaveVmwe.isEmbeded:
                            result += 1

        return result

    def recognizeContinouosandSingleVMWEs(self):
        singleWordExp, continousExp = 0,0
        for mwe in self.vMWEs:
            if len(mwe.tokens) == 1:
                mwe.isSingleWordExp = True
                mwe.isContinousExp = True
                singleWordExp +=1
                continousExp +=1
            else:
                if self.isContinousMwe(mwe):
                    continousExp +=1
        return singleWordExp, continousExp

    def isContinousMwe(self, mwe):
        idxs = []
        for token in mwe.tokens:
            idxs.append(self.tokens.index(token))
        #range = xrange(min(idxs), max(idxs))
        mwe.isContinousExp = True
        for i in range(min(idxs), max(idxs)): #range:
            if i not in idxs:
                mwe.isContinousExp = False
        return mwe.isContinousExp


    def recognizeInterleavingVMWEs(self):
        if len(self.vMWEs) <= 1:
            return 0
        result = 0
        for vmwe in self.vMWEs:
            if vmwe.isEmbeded or vmwe.isInterleaving:
                continue
            for token in vmwe.tokens:
                if len(token.parentMWEs) > 1:
                    for parent in token.parentMWEs:
                        if parent is not vmwe:
                            if parent.isEmbeded:
                                continue
                            parents = sorted(token.parentMWEs, key=lambda VMWE: VMWE.id)
                            if parent != parents[0]:
                                parent.isInterleaving = True
                                result += 1
        return result

    def getCorpusText(self, gold=True):
        if gold:
            mwes = self.vMWEs
        else:
            mwes = self.identifiedVMWEs
        lines = ''
        idx = 1
        for token in self.tokens:
            line = str(idx) + '\t' + token.text + '\t_\t'
            idx += 1
            for mwe in mwes:
                if token in mwe.tokens:
                    if line.endswith('\t'):
                        line += str(mwe.id)
                    else:
                        line += ';' + str(mwe.id)
            if line.endswith('\t'):
                line += '_'
            lines += line + '\n'
        return lines

    def getCorpusTextWithPlus(self):
        goldMwes = self.vMWEs
        predMwes = self.identifiedVMWEs
        lines = ''
        idx = 1
        for token in self.tokens:
            line = str(idx) + '\t' + token.text + '\t_\t'
            idx += 1
            for mwe in goldMwes:
                if token in mwe.tokens:
                    if line.endswith('\t'):
                        line += '+'
                        break

            if line.endswith('\t'):
                line += '_\t'
            else:
                line += '\t'
            for mwe in predMwes:
                if token in mwe.tokens:
                    if line.endswith('\t'):
                        line += '+'
                        break
            if line.endswith('\t'):
                line += '_'
            lines += line + '\n'
        return lines

    def printSummary(self):
        vMWEText = ''
        for vMWE in self.vMWEs:
            vMWEText += str(vMWE) + '\n'
        if len(self.identifiedVMWEs) > 0:
            identifiedMWE = '### Identified MWEs: \n'
            for mwe in self.identifiedVMWEs:
                identifiedMWE += str(mwe) + '\n'
        else:
            identifiedMWE = ''

        return '## Sentence No. ' + str(self.id) + ' - ' + self.sentid + '\n' + self.text + \
               '\n### Existing MWEs: \n' + vMWEText + identifiedMWE

    def __str__(self):

        vMWEText = ''
        for vMWE in self.vMWEs:
            vMWEText += str(vMWE) + '\n'
        if len(self.identifiedVMWEs) > 0:
            identifiedMWE = '### Identified MWEs: \n'
            for mwe in self.identifiedVMWEs:
                identifiedMWE += str(mwe) + '\n\n'
        else:
            identifiedMWE = ''
        featuresInfo = ''

        result = ''
        transition = self.initialTransition
        idx = 0
        while True:
            type = ''
            configuration = ''
            if transition is not None:
                if transition.type is not None:
                    type = transition.type.name
                else:
                    type = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
                configuration = str(transition.configuration)
                if type == 'MERGE':
                    type = '**MERGE**&nbsp;&nbsp;&nbsp;'
                if len(type) == 'SHIFT':
                    type = type + '&nbsp;&nbsp;&nbsp;'
                result += '\n\n' + str(
                    transition.id) + '- ' + type + '&nbsp;&nbsp;&nbsp;' + '>' + '&nbsp;&nbsp;&nbsp;' + configuration + '\n\n'
                if transition.next is None:
                    break
                transition = transition.next
                if len(self.featuresInfo) == 2 and len(self.featuresInfo[1] )> 0:
                    sortedDic = sorted(self.featuresInfo[1][idx].items(), key=operator.itemgetter(0))
                    for item in sortedDic:
                        result += str(item[0]) + ': ' + str(item[1]) + ', '
                idx += 1
            else:        #result += str(self.featuresInfo[1][idx]) + '\n\n'
                break

        # if len(self.featuresInfo) == 2:
        #     labels = self.featuresInfo[0]
        #     features = self.featuresInfo[1]
        #     for x in xrange(0, len(labels)):
        #         featuresInfo += str(x) + '- ' + str(labels[x]) + ' : ' + str(features[x]) + '\n\n'
        return '## Sentence No. ' + str(self.id) + ' - ' + self.sentid + '\n' + self.text + \
               '\n### Existing MWEs: \n' + vMWEText + identifiedMWE \
               + '\n' + result #str(self.initialTransition) + '\n### Features: \n' + featuresInfo


class Token:
    """
        a class used to encapsulate all the information of a sentence tokens
    """

    def __init__(self, position, txt, lemma='', posTag='', abstractPosTag='', morphologicalInfo=[], dependencyParent=-1,
                 dependencyLabel=''):
        try:    # added only for one case of line number 24.1 in the English data
            self.position = int(position)
        except ValueError:  # added only for one case of line number 24.1 in he English data
            self.position = float(position)
        self.text = txt
        # if lemma == '':
        #    self.lemma = Token.wordNetLemmatiser.lemmatize(txt)
        # else:
        self.lemma = lemma
        self.abstractPosTag = abstractPosTag
        self.posTag = posTag
        self.morphologicalInfo = morphologicalInfo
        self.dependencyParent = dependencyParent
        self.dependencyLabel = dependencyLabel
        self.parentMWEs = []

    def setParent(self, vMWE):
        self.parentMWEs.append(vMWE)

    def __str__(self):
        parentTxt = ''
        if len(self.parentMWEs) != 0:
            for parent in self.parentMWEs:
                parentTxt += str(parent) + '\n'

        return str(self.position) + ' : ' + self.text + ' : ' + self.posTag + '\n' + 'parent VMWEs:\t' + parentTxt


class VMWE:
    """
        A class used to encapsulate the information of a verbal multi-word expression
    """

    def __init__(self, id, token=None, type=None, isEmbeded=False, isInterleaving=False, isInTrainingCorpus=0):
        self.id = int(id)
        self.isInTrainingCorpus = isInTrainingCorpus
        self.tokens = []
        self.isSingleWordExp = False
        self.isContinousExp = False
        if token is not None:
            self.tokens.append(token)
        self.type = ''
        if type is not None:
            self.type = type
        self.isEmbeded = isEmbeded
        self.isInterleaving = isInterleaving
        self.isVerbal = True

    def getId(self):
        return self.id

    def addToken(self, token):
        self.tokens.append(token)

    @staticmethod
    def isVerbalMwe(mwe):
        isVerbal = False
        for token in mwe.tokens:
            if token.posTag.startswith('V'):
                isVerbal = True
        return isVerbal

    def __str__(self):
        tokensStr = ''
        for token in self.tokens:
            tokensStr += token.text + ' '
        tokensStr = tokensStr.strip()
        isInterleaving = ''
        if self.isInterleaving:
            isInterleaving = ', Interleaving '
        isEmbedded = ''
        if self.isEmbeded:
            isEmbedded = ', Embedded'
        #isContinousExp =''
        #if self.isContinousExp:
            #isContinousExp = 'Continous'
        inTrainingCorpus = ''
        if self.isInTrainingCorpus != 0:
            inTrainingCorpus = ', ' + str(self.isInTrainingCorpus)
        type = ''
        if self.type != '':
            type = '(' + self.type
            if self.isInTrainingCorpus != 0:
                type += ', ' + str(self.isInTrainingCorpus) + ')'
            else:
                type += ')'
        return str(self.id) + '- ' + '**' + tokensStr + '** ' + type + isEmbedded + isInterleaving

    def getString(self):
        result = ''
        for token in self.tokens:
            result += token.text + ' '
        return result[:-1]

    def getLemmaString(self):
        result = ''
        for token in self.tokens:
            if token.lemma.strip() != '':
                result += token.lemma + ' '
            else:
                result += token.text + ' '
        return result[:-1]

############################################################
