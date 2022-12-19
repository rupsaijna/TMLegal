#python bow_classifyTM.py -f ../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt -b
import sys
sys.path.append('../pyTsetlinMachineParallel/')  #ensure correct relative path to TM code folder
from tm import MultiClassTsetlinMachine

import argparse
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import time
import random
from helperfunctions import *
parser = argparse.ArgumentParser()


maxstorylen=4
parser.add_argument("-f", "--input_file", help = "Original input file", required = True, default = "")
parser.add_argument("-b", "--record_best", help = "Turn on to record as best", required = False, action='store_true')

argument = parser.parse_args()

fname=argument.input_file
head,tail=os.path.split(fname)

fmeta='../featurefiles/'+os.path.split(head)[1]+'_'+tail.replace('.txt','_meta.txt')
fdata='../featurefiles/'+os.path.split(head)[1]+'_'+tail.replace('.txt','_bowfeatures.txt')

folder_name='results/'+sys.argv[0].replace('.py','')+"/"+os.path.split(head)[1]+'_'+tail.replace('.txt','')
if not os.path.exists(folder_name):
	os.makedirs(folder_name)
    
timestr = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(folder_name+'/'+timestr)
folder_name=folder_name+'/'+timestr+'/'
meta_file=folder_name+'meta_details.txt'
result_file=folder_name+'result_details.npz'
global_clause_file=folder_name+os.path.split(head)[1]+'_'+tail.replace('.txt','_global_clauses.csv')
local_clause_file=folder_name+os.path.split(head)[1]+'_'+tail.replace('.txt','_local_clauses.csv')

metadata = {}
with open(fmeta) as f:
    for line in f:
       (key, val) = line.replace('\n','').split('\t')
       metadata[key] = val
df =pd.read_csv(fdata, sep='\t')

datadf=df.loc[df['SubStoryLen']<=maxstorylen]
datadf = datadf.loc[:, (datadf.sum(axis=0)!=0)]

y=datadf['Answer_Numerical'].to_list()
datadf=datadf.drop(['Answer_Numerical'], axis=1)

print('Shape',datadf.shape)

CLASSES=list(set(datadf['Answer']))
gh=metadata['General Headers'].strip('}{').replace("'","").split(',')
datadf=datadf.drop(gh, axis=1)
X=datadf.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, shuffle=True)

NUM_LOCAL_EXAMPLES=5
NUM_CLAUSES=120 #80
T=100
s=5
WEIGHING = True
TRAIN_EPOCHS=10
RUNS=400
NUM_FEATURES=len(X[0])
APPEND_NEGATED=True

acc=np.zeros(RUNS)
train_time=np.zeros(RUNS)
test_time=np.zeros(RUNS)
prec_macro=np.zeros(RUNS)
recall_macro=np.zeros(RUNS)
fscore_macro=np.zeros(RUNS)
prec_micro=np.zeros(RUNS)
recall_micro=np.zeros(RUNS)
fscore_micro=np.zeros(RUNS)

fo=open(meta_file,'w')
fo.write('Question Answering with BoW\n')
fo.write(sys.argv[0]+'\n')
fo.write('\nNum Clauses:'+str(NUM_CLAUSES))
fo.write('\nNum Classes: '+ str(len(CLASSES)))
fo.write('\nT: '+str(T))
fo.write('\ns: '+str(s))
fo.write('\nNum Features: '+ str(NUM_FEATURES)+'\n\n')
fo.write('\nTotal Runs: '+str(RUNS))
fo.write('\nTrain Epochs: '+str(TRAIN_EPOCHS))
fo.write('\nWeighing: '+str(WEIGHING))
fo.write('\nAppend Negated: '+str(APPEND_NEGATED))


tm = MultiClassTsetlinMachine(NUM_CLAUSES, T, s,  weighted_clauses=WEIGHING,append_negated=APPEND_NEGATED)
for r in range(RUNS):
	print('\nepoch:',r)
	start_training = time.time()
	tm.fit(x_train, np.asarray(y_train), epochs=TRAIN_EPOCHS, incremental=True)
	stop_training = time.time()
	train_time[r]= stop_training-start_training
	start_testing = time.time()
	pred=tm.predict(x_test)
	stop_testing = time.time()
	test_time[r]= stop_testing-start_testing
	acc[r] = 100*(pred == y_test).mean()
	
	prf1=precision_recall_fscore_support(pred, y_test, average='macro')
	prec_macro[r]=prf1[0]
	recall_macro[r]=prf1[1]
	fscore_macro[r]=prf1[2]
	prf2=precision_recall_fscore_support(pred, y_test, average='micro')
	prec_micro[r]=prf2[0]
	recall_micro[r]=prf2[1]
	fscore_micro[r]=prf2[2]
	#print('Accuracy, Macro, Micro:',acc[r], prf1, prf2)
	if r>0 and r%5==0:
		np.savez(result_file, accuracy=acc, prec_macro=prec_macro, recall_macro=recall_macro, fscore_macro=fscore_macro,prec_micro=prec_micro, recall_micro=recall_micro, fscore_micro=fscore_micro, training_times= train_time, testing_times=test_time)
print('Accuracy:',acc[-20:].mean(), acc.max())
print('F1:',fscore_macro[-20:].mean(), fscore_macro.max(),fscore_micro[-20:].mean(), fscore_micro.max())

np.savez(result_file, accuracy=acc, prec_macro=prec_macro, recall_macro=recall_macro, fscore_macro=fscore_macro,prec_micro=prec_micro, recall_micro=recall_micro, fscore_micro=fscore_micro, training_times= train_time, testing_times=test_time)
fo.write('\n\nBest result:'+str(acc.max()))
fo.write('\nMean result:'+str(acc.mean()))
fo.close()
print("Meta File at: "+meta_file)
print("Result File at: "+result_file)

####GLOBAL VIEW###
print("\nWriting global Clauses")
fh=metadata['Feature headers'].strip('}{').replace("'","").split(',')
ls=metadata['Possible Answers'].strip('}{').replace("'","").split(',')
wrote_clauses(global_clause_file, NUM_FEATURES, NUM_CLAUSES, tm, fh, ls)

####LOCAL VIEW###
get_random_local_view(x_test,y_test, NUM_LOCAL_EXAMPLES, local_clause_file, tm, global_clause_file, fh, ls)

if argument.record_best:
	print("\nRecording parameters as best")
	bestfile='results/'+sys.argv[0].replace('.py','')+"/"+os.path.split(head)[1]+'_'+tail.replace('.txt','')+'/bestfile.txt'
	fo=open(bestfile,'a+')
	fo.write("\nTime marker:"+timestr)
	fo.write('Task 1 BOW classification')
	fo.write(sys.argv[0]+'\n')
	fo.write('\nNum Clauses:'+str(NUM_CLAUSES))
	fo.write('\nNum Classes: '+ str(len(CLASSES)))
	fo.write('\nT: '+str(T))
	fo.write('\ns: '+str(s))
	fo.write('\nNum Features: '+ str(NUM_FEATURES)+'\n\n')
	fo.write('\nTotal Runs: '+str(RUNS))
	fo.write('\nTrain Epochs: '+str(TRAIN_EPOCHS))
	fo.write('\nWeighing: '+str(WEIGHING))
	fo.write('\nAppend Negated: '+str(APPEND_NEGATED))
	fo.write('\n\nBest result:'+str(acc.max()))
	fo.write('\nMean result:'+str(acc.mean()))
	fo.write('\n-------------------------------\n\n')
	fo.close()
	print("Parameters recorded as best to: "+bestfile)
