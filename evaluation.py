import os

def labels2Parsemetsv(labels, mainTest, predOut):
    """
    labels: predicted labels
    mainTest: the file that contains predicted labels should be matched with gold file
    predOut: output

    This function is used to convert IOB labeling back to the Parseme format for evaluation.
    """
    with open(predOut, 'w') as predFile:
        with open(mainTest) as bt:
            lines = bt.readlines()
            sentIdx = -1
            i=0
            for line in lines:
                i+=1
                if line == '\n' or line.startswith(' #') or line.startswith('#'):
                    predFile.write(line)
                    wrdIdx = 0
                    mweNum = 0
                    mwe = {}
                    continue
                if len(line)>0 and line.endswith('\n'):
                    line = line[:-1]
                if line.startswith("1\t"):
                    sentIdx += 1
                    wrdIdx = 0
                    mweNum = 0
                    mwe = {}
                lineParts = line.split('\t')

                if '-' in lineParts[0] or '.' in lineParts[0]:
                    predFile.write("\t".join(lineParts[0:-1])+"\t"+"*"+"\n")
                    continue
                if not lineParts[0].isdigit():
                    print("ERROR: What is this line:",line)
                    predFile.write(line+"\n")
                    continue
                if sentIdx<len(labels):
                    if type(labels[sentIdx])== 'int':
                        print("ERRor (sent index)", sentIdx, labels[sentIdx])
                        print(line)
                    try:
                        if labels[sentIdx][wrdIdx] == 'O':
                            predFile.write("\t".join(lineParts[0:-1])+"\t"+"*"+"\n")
                            wrdIdx += 1
                            continue
                        elif str(labels[sentIdx][wrdIdx]).startswith('B_') or str(labels[sentIdx][wrdIdx]).startswith('I_'):
                            diffTags = labels[sentIdx][wrdIdx].split(';')
                            if diffTags[0].startswith('B_'):
                                mweNum += 1
                                tag = str(mweNum)+":" + diffTags[0][2:]
                                mwe[diffTags[0][2:]] = mweNum  # A MWE type might not be unique in a sentence, but the reason that
                                                               # I use a dictionary like this is that I need to keep track of the
                                                               # most recent MWE type. We don't consider and we probably don't have
                                                               # two intervening LVCs 
                            elif diffTags[0].startswith('I_'):
                                if diffTags[0][2:] in mwe:
                                    tag = str(mwe[diffTags[0][2:]])
                                else:          # An I tag should not exist without B, so we filter it out. 
                                    tag = "*"   
                            for idiff in range(1,len(diffTags)):
                                if diffTags[idiff].startswith('B_'):
                                    mweNum += 1
                                    if tag != "*":
                                        tag = tag + ';' + str(mweNum)+":" + diffTags[idiff][2:]
                                    else:
                                        tag = str(mweNum)+":" + diffTags[idiff][2:]
                                    mwe[diffTags[idiff][2:]] = mweNum
                                elif diffTags[idiff].startswith('I_'):
                                    if diffTags[idiff][2:] in mwe:   
                                        if tag!="*" and not str(mwe[diffTags[idiff][2:]]) in tag:
                                                        # this 'not' is to make sure that the tags before and after ';'
                                                        # do not belong to the same mwe type
                                            tag = tag + ";" + str(mwe[diffTags[idiff][2:]])
                                        else:
                                            tag = str(mwe[diffTags[idiff][2:]])
                            predFile.write("\t".join(lineParts[:-1])+"\t"+ tag +"\n")
                            wrdIdx +=1
                            continue
                        else:       # This happens when something is labeled as <PADLABEL>!
                            predFile.write("\t".join(lineParts[0:-1])+"\t"+"*"+"\n")
                            wrdIdx += 1
                            continue
                    except TypeError:
                        print("ERRor (sent index)", sentIdx, labels[sentIdx])
                        print(line)
 
                else:
                    print("sent ids greater than labels!")
            
            if sentIdx<len(labels)-1:
                print("ERROR: This prediction is not for this file", sentIdx, len(labels))
                predFile.write("ERROR\n")
