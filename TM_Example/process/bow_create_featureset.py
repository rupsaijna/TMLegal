#from original data, take all words and create BoW

import numpy as np
import sys
import pandas as pd
import os
generalheaders=['StoryNum','SubStory','Question','Answer','SubStoryLen']

fname=sys.argv[1]
head,tail=os.path.split(fname)

fmeta='../featurefiles/'+os.path.split(head)[1]+'_'+tail.replace('.txt','_meta.txt')
fdata='../featurefiles/'+os.path.split(head)[1]+'_'+tail.replace('.txt','_bowfeatures.txt')

f=open(fname,'r')
data=f.readlines()
f.close()


combineddata=[]
word_set_sentences= []
word_set_questions= []
lind=0
storynum=0
for line in data:
	line=line.replace('\n','')
	line=line.split('\t')
	sentence=line[0].strip().split(' ')
	sentindex=int(sentence[0])
	sentence=' '.join(sentence[1:])
	if sentindex==1:
		storynum+=1
		storylen=0
		storysentences=[]
		storysentences+=[sentence]
		word_set_sentences+=sentence.replace('.','').split(' ')
		storylen+=1
	else:
		if len(line)==1:
			storysentences+=[sentence]
			word_set_sentences+=sentence.replace('.','').split(' ')
			storylen+=1
		else:
			answer=line[1]
			word_set_questions+=sentence.replace('?','').split(' ')
			combineddata+=[[storynum,' '.join(storysentences), sentence, answer, storylen]]
	lind+=1

		

word_set_sentences=list(set(word_set_sentences))
word_set_questions=list(set(word_set_questions))
df = pd.DataFrame(combineddata, columns=generalheaders)
answers_set=list(set(df['Answer'].values))

maxlen=df["SubStoryLen"].max()
featureheaders=[]
for n in range(1,maxlen+1):
	featureheaders+=['s_'+str(n)+'_'+w for w in word_set_sentences]
featureheaders+=['q_'+w for w in word_set_questions]

df[featureheaders] = pd.DataFrame([[0]*len(featureheaders)], index=df.index)
df['Answer_Numerical'] = pd.DataFrame([[-99]], index=df.index)
for ind, row in df.iterrows():
	print(ind)
	sentences=[s.strip() for s in row['SubStory'].split('.')][:-1]
	sind=0
	for sent in sentences:
		sind+=1
		words=sent.split(' ')
		for w in words:
			df.at[ind,'s_'+str(sind)+'_'+w ]=1
	words= row['Question'].replace('?','').strip().split(' ')
	for w in words:
		df.at[ind,'q_'+w ]=1
	df.at[ind,'Answer_Numerical']=int(answers_set.index(row['Answer']))
	

fm=open(fmeta,'w')
fm.write('Source File\t'+fname)
fm.write('\nProcessing\tBoW')
fm.write('\nNum General Headers\t'+str(len(generalheaders)))
fm.write('\nGeneral Headers\t'+','.join(generalheaders))
fm.write('\nNum Feature headers\t'+str(len(featureheaders)))
fm.write('\nFeature headers\t'+','.join(featureheaders))
fm.write('\nMax Story Len\t'+str(maxlen))
fm.write('\nNum Stories\t'+str(df.tail(1)['StoryNum'].values[0]))
fm.write('\nNum Possible Answers\t'+str(len(answers_set)))
fm.write('\nPossible Answers\t'+str(','.join(answers_set)))
fm.close()

df.to_csv(fdata, sep='\t', index=False)

print("Features saved at:", fdata)
