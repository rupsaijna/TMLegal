Requirements Tested With

Python 3.6.8

pandas 1.1.5

sklearn 0.21.3

numpy 1.19.5

###################################################################################

Setting up
-------------------------------------------------------------------------------------------------------------------------
$mkdir data

$mkdir featurefiles

$mkdir results

>If we don't want to see local clauses, PyTsetlinMachineCUDA works fastest. Else we use the pyTsetlinMachineParallel code contained in this folder.


$pip install PyTsetlinMachineCUDA

$pip install pycuda

###################################################################################

Get the Data
-------------------------------------------------------------------------------------------------------------------------

$cd data

$wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz

$tar -xvzf tasks_1-20_v1-2.tar.gz


###################################################################################

Create features
-------------------------------------------------------------------------------------------------------------------------

$cd process

$python bow_create_featureset.py ../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt

###################################################################################

Classify and see global clauses
-------------------------------------------------------------------------------------------------------------------------

$cd classify

$python bow_classifyTM.py -f ../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt -b

>-f required, takes original data file path
>
>-b optional, records current parameters as best performance(good for record keeping)

###################################################################################

Classify and see local clauses (as well as global clauses)
-------------------------------------------------------------------------------------------------------------------------

$cd pyTsetlinMachineParallel

$make

$cd ../classify_with_local_view

$python bow_classifyTM.py -f ../data/tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_train.txt -b

>-f required, takes original data file path
>
>-b optional, records current parameters as best performance (good for record keeping)

###################################################################################

Other Information
-------------------------------------------------------------------------------------------------------------------------

Global clause file has the following columns : ClauseNum,Clause,p/n,class

1. ClauseNum: Clause Number

2. Clause: Clause Content

3. p/n: Clause Polarity (Positive or Negative)

4. class: Clause Class


Local clause file has the following columns: Example,Class,Clause,Cl.Val, Example_BoW, ClassName, ClauseText, CorrectLabel

1. Example:Test Sample number

2. Class: Numerical Class

3. Clause: Clause Number

4. Cl.Val: Votes received by Clause for Class

5. Example_BoW: BoW corresponding to Test Sample

6. ClassName: Textual Class

7. ClauseText: Clause Content

8. CorrectLabel: 'Gold' if Class corresponds to gold standard and/or 'Predicted' if Class corresponds to final prediction

-------------------------------------------------------------------------------------------------------------------------

Tsetlin Machine Controllable Parameters:

1. NUM_CLAUSES: number of clauses per class, Int

2. T: Thrsehold, Int

3. s: sensitivity, Float

4. WEIGHING: Use Weighted Clauses, Boolean

5. TRAIN_EPOCHS: Training rounds per Run, Int

6. APPEND_NEGATED: Use negated features, Boolean


Experiment Parameters:

1. NUM_LOCAL_EXAMPLES: Number of Examples to print for seeing Local Clauses

2. RUNS=Individual Runs, Int

#################################################################################

## References

Relational Tsetlin Machine : Saha, R., Granmo, O. C., Zadorozhny, V. I., & Goodwin, M. (2021). A Relational Tsetlin Machine with Applications to Natural Language Understanding. arXiv preprint arXiv:2102.10952.
https://arxiv.org/abs/2102.10952

PyTsetlinMachineCUDA : K. Darshana Abeyrathna, Bimal Bhattarai, Morten Goodwin, Saeed Gorji, Ole-Christoffer Granmo, Lei Jiao, Rupsa Saha, and Rohan K. Yadav (2020). Massively Parallel and Asynchronous Tsetlin Machine Architecture Supporting Almost Constant-Time Scaling. arXiv preprint arXiv:2009.04861, 
https://arxiv.org/abs/2009.04861
Code at [https://github.com/cair/PyTsetlinMachineCUDA]

pyTsetlinMachineParallel: Code at [https://github.com/cair/pyTsetlinMachineParallel]

Original Paper Source for Data:

[Weston, J., Bordes, A., Chopra, S., Rush, A. M., van Merriënboer, B., Joulin, A., & Mikolov, T. (2015). Towards ai-complete question answering: A set of prerequisite toy tasks. arXiv preprint arXiv:1502.05698.]
(https://arxiv.org/pdf/1502.05698.pdf)


## Licence

Copyright (c) 2021 Rupsa Saha

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
