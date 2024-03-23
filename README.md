# Python Vulnerability Detection with Named Entity Recognition

The full background and process of this work is shown in: https://doi.org/10.1016/j.cose.2024.103802
_______________________________
Vulnerabilities within source code have grown over the last 20 years to become a common threat to systems and networks. As the implementation of open-source software continues to develop, more unknown vulnerabilities will exist throughout system networks. This research proposes an enhanced vulnerability detection method specific to Python source code that utilizes pre-trained, BERT-based transformer models to apply tokenization, embedding, and named entity recognition (a natural language processing technique). The use of named entity recognition not only allows for the detection of potential vulnerabilities, but also for the classification of different vulnerability types. This research uses the publicly available CodeBERT, RoBERTa, and DistilBERT models to fine-tune for the downstream task of token classification for six different common weakness enumeration specifications. The results achieved in this research outperform previous Python-based vulnerability detection methods and demonstrate the effectiveness of applying named entity recognition to enhance the overall research into Python source code vulnerabilities.
______________________________

PyVulNER.ipynb includes all information necessary from complete data collection to model testing.
