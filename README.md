## LLM-Resilient Bibliometrics: Factual Consistency Through Entity Triplet Extraction
This github repository provides the code that belongs to the paper "LLM-Resilient Bibliometrics: Factual Consistency Through Entity Triplet Extraction". The code provides the full pipeline from raw arXiv pdf's to processed entity triplets of the shape (subject, predicate, object).

### Structure :books:
-- **src** \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```load_data.py```: This file loads the pdf's and converts them to text files \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```preprocessing.py```: This file preprocesses the text files \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```extract_claims.py```: This file extracts the core claims from the text files \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```extract_triplets.py```: This file extracts the triplets from the text files \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```clustering.py```: This file clusters the triplets based on the subject, object, or both \
&nbsp;&nbsp;&nbsp;&nbsp;|--- ```helpers.py```: This file provides a helper function for logging \
\
-- ```requirements.txt```: File with environment requirements

### Requirements :mag:

#### Environment
The file ```requirements.txt``` contains the requirements needed, which are compatible with python 3.11.7. Using the following code snippet, an environment can be created:

```
conda create -name <env_name> python=3.11.7 
pip install -r requirements.txt
```

#### Data
Before using the code, you need to load data and the claim extraction model.

* The code is designed for the extraction of triplets from arXiv papers. These articles are publicly available in a Google Cloud bucket, for more information read [this Kaggle page](https://www.kaggle.com/datasets/Cornell-University/arxiv).
* The claim extraction model originates from [the paper of Wei et al. "ClaimDistiller: Scientific Claim Extraction with Supervised
Contrastive Learning"]([https://www.semanticscholar.org/paper/Extracting-Core-Claims-from-Scientific-Articles-Jansen-Kuhn/acb6fd4058b3c6ce491d5cde499d7733909bc8a9](https://ceur-ws.org/Vol-3451/paper11.pdf)). The trained WC-BiLSTM model is available in [this drive](https://drive.google.com/drive/folders/1KnaMKNVDYrydH0GrvBTM6lW08M1aGACm?usp=sharing). This model should be placed in a folder ```"models/claim_model"``` in the root directory.

### Usage of the code :memo:
The code is designed for the extraction of triplets from arXiv papers. There are several things to take into account when using the code:

* First define your set of _target papers_, from which you want to extract triplets. Put these papers, together with the metadata in a folder. Now you are ready to load the data with ```load_data.py``` and preprocess the text with ```preprocessing.py```.
* Now the claims can be extracted with ```claim_extraction.py```, make sure that the claim extraction model is in the correct place. Afterwards, the triplets can be extracted with ```triplet_extraction.py```.
* Finally, you can cluster the triplets with ```clustering.py``` based on the embedding of the subject, object or of both. 
