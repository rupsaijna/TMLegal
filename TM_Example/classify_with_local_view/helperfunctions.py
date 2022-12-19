import numpy as np
import pandas as pd
import random
import os

def get_random_local_view(X_test,y_test, num, local_clause_file, tm, clause_file, featureheaderset, labels_set):
    print("Going Local")
    indx=random.randint(num,len(X_test))
    temp_X_test=X_test[indx-num:indx]
    temp_y_test=y_test[indx-num:indx]
    temp_X_test_sent=[]
    for l in range(len(temp_X_test)):
        temp_sent=[]
        line=temp_X_test[l]
        for ft in range(len(line)):
            if line[ft]==1:
                temp_sent.append(featureheaderset[ft])
        temp_X_test_sent.append(' '.join(temp_sent))

    if os.path.exists(local_clause_file):
        print('overwriting previous local file'+local_clause_file)
        os.remove(local_clause_file)
    fo=open(local_clause_file,'w')
    fo.write('Example Class Clause Cl.Val\n')
    fo.close()
    res=tm.predict_and_printlocal(temp_X_test, local_clause_file)

    local_clauses=pd.read_csv(local_clause_file,sep=' ')
    for ts in range(len(temp_X_test_sent)):
        for ind,row in local_clauses.iterrows():
            if row['Example']==ts:
                local_clauses.loc[local_clauses.index[ind], 'Example_BoW']=temp_X_test_sent[ts]
                local_clauses.loc[local_clauses.index[ind], 'ClassName']=labels_set[int(row['Class'])]
    all_clauses=pd.read_csv(clause_file,sep='\t')
    for ind,row in local_clauses.iterrows():
        classname=row['ClassName']
        clauseid=int(row['Clause'])
        clausetext=all_clauses[(all_clauses['ClauseNum']==clauseid) & (all_clauses['class']==classname) ]['Clause'].values
        local_clauses.loc[local_clauses.index[ind], 'ClauseText']=clausetext
        star=''
        if row['Class']==temp_y_test[row['Example']]:
            star+='Gold'
        if row['Class']==res[row['Example']]:
            star+='Predicted'
        local_clauses.loc[local_clauses.index[ind], 'CorrectLabel']=star

    local_clauses=local_clauses.sort_values(by=['Example', 'Class'])

    local_clauses.to_csv(local_clause_file, sep='\t', index=False)
    print('Local Clauses written to:'+local_clause_file)
    
    
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
                #    this_clause+='#'+featureheaderset[f]+';'
            this_clause+='\t'+clause_type+'\t'+str(labels_set[cur_cls])    
            fout_c.write(str(this_clause)+'\n')
    fout_c.close()

    print('Global Clauses written at :'+ clause_file)
