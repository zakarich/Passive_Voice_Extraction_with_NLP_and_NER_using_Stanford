import nltk
from nltk import Tree
def isPassive(tokens,words):
    beforms = ['am', 'is', 'are', 'been', 'was', 'were', 'be', 'being']               
    # all forms of "be"
    aux = ['do', 'did', 'does', 'have', 'has', 'had']                                  
    # NLTK tags "do" and "have" asverbs, which can be misleading in the following section.
    tags = [i[1] for i in tokens]
    if tags.count('VBN') == 0:                                                            
        # no paste participle, no passive voice.
        return False
    elif tags.count('VBN') == 1 and 'been' in words:                                    
        # one VBN (Paste Participle) as "been", still no passive voice.
        return False
    else:
        pos = [i for i in range(len(tags)) if tags[i] == 'VBN' and words[i] != 'been']  
        # gather all the VBN (verb past participle) that are not "been".
        for end in pos:
            chunk = tags[:end]
            start = 0
            for i in range(len(chunk), 0, -1):
                last = chunk.pop()
                if last == 'NN' or last == 'PRP':
                    start = i                                                             
                    # get the chunk between VBN(Paste Particple) and the previous NN (Noun, singular or mass) or PRP (Personal Pronoun) (which in most cases are subjects)
                    break
            sentchunk = words[start:end]
            tagschunk = tags[start:end]
            verbspos = [i for i in range(len(tagschunk)) if tagschunk[i].startswith('V')] 
            # get all the verbs in between
            if verbspos != []:                                                            
                # if there are no verbs in between, it's not passive
                for i in verbspos:
                    if sentchunk[i].lower() not in beforms and sentchunk[i].lower() not in aux:  
                        # check if they are all forms of "be" or auxiliaries such as "do" or "have".
                        break
                else:
                    return True
    return False
    

def clauser(sent):
    # \n is placed to indicate EOL (End of Line) 
    t = nltk.Tree.fromstring(sent)
    #print t
    subtexts = []
    for subtree in t.subtrees():
        if subtree.label()=="S"or subtree.label()=="SBAR" or subtree.label()=="SBARQ" :
            #print subtree.leaves()
            subtexts.append(' '.join(subtree.leaves()))
    #print subtexts
    presubtexts = subtexts[:]       # ADDED IN EDIT for leftover check
    for i in reversed(range(len(subtexts)-1)):
        try:
            subtexts[i] = subtexts[i][0:subtexts[i].index(subtexts[i+1])]
        except: 
            break
    #for text in subtexts:
        #print(text)
        # ADDED IN EDIT - Not sure for generalized cases
    #try:
        #if len(presubtexts) > 1:
            #leftover = presubtexts[0][presubtexts[0].index(presubtexts[1])+len(presubtexts[1]):]
            #subtexts.append(leftover)
    #except : 
        #break
    return subtexts    
    

def TreeCoreNLP(text):
    from pycorenlp import StanfordCoreNLP
    stanford = StanfordCoreNLP('http://localhost:8811')
    output = stanford.annotate(text, properties={'annotators': 'tokenize,ssplit,pos,depparse,parse', 'outputFormat': 'json'})
    return output['sentences'][0]['parse']

def ExtractPhrases( myTree, phrase):
    myPhrases = []
    if (myTree.label() == phrase):
        myPhrases.append( myTree.copy(True) )
    for child in myTree:
        if (type(child) is Tree):
            list_of_phrases = ExtractPhrases(child, phrase)
            if (len(list_of_phrases) > 0):
                myPhrases.extend(list_of_phrases)
    return myPhrases