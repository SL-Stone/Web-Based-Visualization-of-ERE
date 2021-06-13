from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import torch
import torch.nn as nn
import json
import random
from transformers import BertTokenizer, BertModel
from . import model

device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
p='C:/Users/Administrator/Desktop/毕设/Codes/'

def index(request):
    return render(request, 'root.html')

alresults=[]
eresults=set()
rresults=set()
s=""
def get_example(request):
    context = {}
    s = ['After graduating in 1962, she continued her studies in Lausanne, Switzerland.', 
    'As China becomes a major economic and military power and its diplomacy becomes more assertive, Beijing is also working harder at winning friends and influencing people.',
    'Mr. Bush is scheduled to meet President Vladimir V. Putin of Russia outside Moscow on Sunday night.',
    'Nokia, the world\'s largest mobile phone company, put an end to succession questions on Monday when it named one of its top executives, Olli-Pekka Kallasvuo, to replace its longtime chief, Jorma Ollila.',
    ]   
    sentence=s[random.randint(0,4)]
    context['s']=sentence
    return render(request, 'root.html', context)
    


def extract(request):
    global alresults
    global eresults
    global rresults
    global s
    context = {}
    sentence = request.GET.get('sentence')
    
    rel_dict_path = p+'data/NYT/rel2id.json'
    id2rel, _ = json.load(open(rel_dict_path))
    id2rel = {int(i): j for i, j in id2rel.items()}
    Input_size = 60
    num_layers = 1
    CHECKPOINT = torch.load(p+'记录/程序/paramsAtt1_34.pkl',map_location=device)
    sub_model = model.subject_model(Input_size,num_layers)
    sub_model.load_state_dict(CHECKPOINT['sub_state_dict'])
    obj_model = model.object_model(len(id2rel),Input_size,num_layers)
    obj_model.load_state_dict(CHECKPOINT['obj_state_dict']) 
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert_model = BertModel.from_pretrained('D:/Bert-PyTorch/bert-base-cased')
    bert_model.eval()

    context['s']=sentence
    results= model.extract_triples(sub_model, obj_model, tokenizer, bert_model, sentence, id2rel)
    
    
    alresults=results
    eresults=set()
    rresults=set()
    for x in results:
        eresults.add(x[0])
        rresults.add(x[1])
        eresults.add(x[2])   
    s=sentence
    context['eresults']=eresults
    context['rresults']=rresults
    context['results']=results
    return render(request, 'reltriples.html', context)
    
def fix_rel(request):
    global alresults
    global s
    context = {}
    
    rel = request.GET.get('rel')
    rel_triple=[]
    for x in alresults:
        if x[1]==rel:
            rel_triple.append(x)
            
    context['s']=s
    context['eresults']=eresults
    context['rresults']=rresults
    context['results']=rel_triple
    return render(request, 'reltriples.html', context)
    
def fix_entity(request):
    global alresults
    global s
    context = {}
    
    entity = request.GET.get('entity')
    en_triple=[]
    for x in alresults:
        if x[0]==entity or x[2]==entity:
            en_triple.append(x)
    context['s']=s
    context['eresults']=eresults
    context['rresults']=rresults
    context['results']=en_triple
    return render(request, 'reltriples.html', context)
    