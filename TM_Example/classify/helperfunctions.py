import numpy as np
import pandas as pd
import random
import os

	
def wrote_clauses(clause_file, NUM_FEATURES, NUM_CLAUSES, tm, featureheaderset, labels_set):
	fout_c=open(clause_file,'w')
	fout_c.write('ClauseNum\tClause\tp/n\tclass\n')
	feature_vector=np.zeros(NUM_FEATURES*2)
	for cur_cls in range(len(labels_set)):
		for cur_clause in range(NUM_CLAUSES):
			if cur_clause%2==0:
				clause_type='positive'
			else:
				clause_type='negative'
			this_clause=str(cur_clause)+'\t'
			for f in range(0,NUM_FEATURES):
				action_plain = tm.ta_action(int(cur_cls), cur_clause, f)
				action_negated = tm.ta_action(int(cur_cls), cur_clause, f+NUM_FEATURES)
				feature_vector[f]=action_plain
				feature_vector[f+NUM_FEATURES]=action_negated
				if action_plain==1:
					this_clause+=featureheaderset[f]+';'
				#if action_negated==1:
				#	this_clause+='#'+featureheaderset[f]+';'
			this_clause+='\t'+clause_type+'\t'+str(labels_set[cur_cls])	
			fout_c.write(str(this_clause)+'\n')
	fout_c.close()

	print('Global Clauses written at :'+ clause_file)
