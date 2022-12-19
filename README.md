# Quick information

* bert.ipynb is used for both RoBERTa and BERT model.
* cnnbilstm.ipynb is used for the CNN-BILSTM model.
* fasttext.ipynb is used for the FastText model.
* TM_Example/binary_tm.ipynb is used for the CUDA version of the TM without the local clauses
* TM_Example/binary_tm_local_clauses.ipynb is used for the non-CUDA version of the TM with the local clauses.

Inside each of the files listed above, will be a short explanation for each of the files, and what is necessary in order to get it up and running.

All models that uses the SpaCy tokenizer must have the spacy word vocabulary downloaded. This can be done in this way:

> python3 -m spacy download "en_core_web_sm"