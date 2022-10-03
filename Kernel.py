from nltk import Tree
import nltk
import re
from pycorenlp import *
import pandas as pd
import requests
import json
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import time
import datetime
import os.path
import Kernel as ker 

nlp=StanfordCoreNLP("http://localhost:8811/")
def clauser(sent):
    # \n is placed to indicate EOL (End of Line) 
    t = Tree.fromstring(sent)
    #print t
    subtexts = []
    for subtree in t.subtrees():
        if subtree.label()=="S"or subtree.label()=="SBAR":
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
    try:
        leftover = presubtexts[0][presubtexts[0].index(presubtexts[1])+len(presubtexts[1]):]
        subtexts.append(leftover)
    except : 
        print('somethings wrong!!!!')
    return subtexts
##########################################################################
#get_verbs_text:
def get_verb_sent(sent):
    try:
        parser = nlp.annotate(sent, properties={"annotators":"parse","outputFormat": "json"})
        t = nltk.tree.ParentedTree.fromstring(parser["sentences"][0]["parse"])
    except:
        exception = []
        print('something wrong at this sent : '+sent+' in this article : ' )
        #exception.append(sent)
    verb_phrases = []
    num_children = len(t)
    num_VP = sum(1 if t[i].label() == "VP" else 0 for i in range(0, num_children))

    if t.label() != "VP":
        for i in range(0, num_children):
            if t[i].height() > 2:
                verb_phrases.extend(get_verb_sent(t[i]))
    elif t.label() == "VP" and num_VP > 1:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                if t[i].height() > 2:
                    verb_phrases.extend(get_verb_sent(t[i]))
    else:
        verb_phrases.append(' '.join(t.leaves()))

    return verb_phrases
##########################################################################    
#get_verbs_tree:
def get_verb_phrases(t):
    verb_phrases = []
    num_children = len(t)
    num_VP = sum(1 if t[i].label() == "VP" else 0 for i in range(0, num_children))

    if t.label() != "VP":
        for i in range(0, num_children):
            if t[i].height() > 2:
                verb_phrases.extend(get_verb_phrases(t[i]))
    elif t.label() == "VP" and num_VP > 1:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                if t[i].height() > 2:
                    verb_phrases.extend(get_verb_phrases(t[i]))
    else:
        verb_phrases.append(' '.join(t.leaves()))

    return verb_phrases

##########################################################################    
def get_pos(t):
    vp_pos = []
    sub_conj_pos = []
    num_children = len(t)
    children = [t[i].label() for i in range(0,num_children)]

    flag = re.search(r"(S|SBAR|SBARQ|SINV|SQ)", ' '.join(children))

    if "VP" in children and not flag:
        for i in range(0, num_children):
            if t[i].label() == "VP":
                vp_pos.append(t[i].treeposition())
    elif not "VP" in children and not flag:
        for i in range(0, num_children):
            if t[i].height() > 2:
                temp1,temp2 = get_pos(t[i])
                vp_pos.extend(temp1)
                sub_conj_pos.extend(temp2)
    # comment this "else" part, if want to include subordinating conjunctions
    else:
        for i in range(0, num_children):
            if t[i].label() in ["S","SBAR","SBARQ","SINV","SQ"]:
                temp1, temp2 = get_pos(t[i])
                vp_pos.extend(temp1)
                sub_conj_pos.extend(temp2)
            else:
                sub_conj_pos.append(t[i].treeposition())

    return (vp_pos,sub_conj_pos)

##########################################################################    
# get all clauses
def get_clause_list(i,sent):
    try:
        parser = nlp.annotate(sent, properties={"annotators":"parse","outputFormat": "json"})
        sent_tree = nltk.tree.ParentedTree.fromstring(parser["sentences"][0]["parse"])
    except:
        exception = []
        print('something wrong at this sent : '+sent+' in this article : '+str(i) )
        #exception.append(sent)
        return [sent]
    clause_level_list = ["S","SBAR","SBARQ","SINV","SQ"]
    clause_list = []
    sub_trees = []
    # break the tree into subtrees of clauses using
    # clause levels "S","SBAR","SBARQ","SINV","SQ"
    for sub_tree in reversed(list(sent_tree.subtrees())):
        if sub_tree.label() in clause_level_list:
            if sub_tree.parent().label() in clause_level_list:
                continue
            if (len(sub_tree) == 1 and sub_tree.label() == "S" and sub_tree[0].label() == "VP"
                and not sub_tree.parent().label() in clause_level_list):
                continue
            sub_trees.append(sub_tree)
            del sent_tree[sub_tree.treeposition()]
    # for each clause level subtree, extract relevant simple sentence
    for t in sub_trees:
        # get verb phrases from the new modified tree
        verb_phrases = get_verb_phrases(t)
        # get tree without verb phrases (mainly subject)
        # remove subordinating conjunctions
        vp_pos,sub_conj_pos = get_pos(t)
        for i in vp_pos:
            try:            
                del t[i]
            except:
                continue
                #print('something wrong!!!!!')
        for i in sub_conj_pos:
            try:
                del t[i]
            except:
                continue
                #print('something wrong!!!!!')

        subject_phrase = ' '.join(t.leaves())
        # update the clause_list
        for i in verb_phrases:
            clause_list.append(subject_phrase + " " + i)
    return clause_list
##########################################################################    
def scrapeHTML(url,category):
    punctuations = '''!()-[]{}:''"“”''\<>/?@#$%^&*_~''' ##i'll use it right now in the next steps 
    text = ""
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page,"html") #taking the code html of the page
    heading = soup.h1.string #takin the balise <h1> that contains title
    contents = soup.find_all("p") #taking all Paragraphs from the page
    #auth = soup.findAll("p", {"class": "css-1nuro5j e1jsehar1"})
    #authors = auth.find_all("span")
    for content in contents:
        text = text+str(content.string)+" "
    heading = re.sub(',' , ' ', heading)
    heading = re.sub(';' , ' ', heading)
    #some pre-processing for our content
    text = re.sub(',' , ' ', text)
    text = re.sub(';' , ' ', text)
    text = re.sub('Advertisement Supported by' , ' ', text)
    text = re.sub('Advertisement','',text)
    text = re.sub('None',' ',text)
    text = re.sub('Mr.','Mr ',text)
    text = re.sub('-', ' ',text)
    text = re.sub('—',' ',text)
    text = re.sub('_',' ',text)
    text = re.sub('F.B.I.','FBI',text)
    text = re.sub('U.S.','US',text)
    text = re.sub('C.S.I.','CSI',text)
    text = re.sub('C.I.A.','CIA',text)
    text = re.sub('S.W.A.T.','SWAT',text)
    text = re.sub('F.B.I','FBI',text)
    text = re.sub('U.S','US',text)
    text = re.sub('C.S.I','CSI',text)
    text = re.sub('C.I.A','CIA',text)
    text = re.sub('S.W.A.T','SWAT',text)
    text = re.sub('  ' , ' ', text)
    for x in text.lower(): 
        if x in punctuations: 
            text = text.replace(x, " ")
    Page = {'URL':url, 'Title':heading, 'Text': text, 'Category':category}
    return Page
##########################################################################    
def scraping(query,filename):

    Pages = pd.DataFrame(columns = ['URL','Title','Text','Category'])


    api_key ="t1pWm68uulpnBM80D61LopROQMyh9epG"
    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json?api-key="+api_key+"&q="+query
    response = requests.get(url)
    response_data = response.content
    json_data =response_data.decode('utf8') # converting data to string
    data = json.loads(json_data);
    s = json.dumps(data, indent = 2, )
    data_dump = data['response']['docs']
    for i in data_dump:
        print(i['web_url'])
        try:
            Pages = Pages.append(scrapeHTML(i['web_url'],filename.replace('.csv','')), ignore_index = True)
        except:
            continue;
    
    my_file = os.path.isfile('articles/'+filename)
    if(my_file):
        old = pd.read_csv('articles/'+filename)
        upd = old.append(Pages)
    else:
        upd = pd.DataFrame(Pages)
    
    upd = upd.drop_duplicates(subset = ['URL'],keep = 'first')
    upd.to_csv('articles/'+filename,index = False)
##########################################################################    

