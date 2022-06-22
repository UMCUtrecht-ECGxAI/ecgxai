<div align="center">    
 
# ECGxAI: Explainable AI for the electrocardiogram     

[![CausalCNN](https://img.shields.io/badge/conference-CausalCNN-blue.svg)](https://moody-challenge.physionet.org/2020/papers/253.pdf)
[![FactorECG](https://img.shields.io/badge/preprint-FactorECG-red.svg)](https://www.medrxiv.org/content/10.1101/2022.01.04.22268759v1)
[![DoubleResidual](https://img.shields.io/badge/conference-qLST-blue.svg)](https://arxiv.org/pdf/2111.07386.pdf)
[![PLN](https://img.shields.io/badge/paper-PLN-purple.svg)](https://www.ahajournals.org/doi/full/10.1161/CIRCEP.120.009056)
   
</div>
 
## Description   
This repository accompanies papers from the *Explainable AI for the ECG (ECGxAI)* research group at the [UMC Utrecht](https://www.umcutrecht.nl/nl) and contains an installable python package to train and evaluate explainable deep learning methods for the analysis of (12-lead) electrocardiograms (ECGs). The package is based on [Pytorch Lightning](https://www.pytorchlightning.ai/), is work-in-progress and new functionalities will be added along the way.

The first version of this package provides functionalities to train the CausalCNN architecture as proposed in [our Computing in Cardiology conference paper](https://moody-challenge.physionet.org/2020/papers/253.pdf) and the DoubleResidual architecture as proposed in our [Machine Learning for Health (ML4H) 2021 conference paper](https://arxiv.org/pdf/2111.07386.pdf). Moreover, it provides functionality to train an explainable deep learning pipeline for ECGs (the [FactorECG](https://www.medrxiv.org/content/10.1101/2022.01.04.22268759v1)), that uses a variational auto-encoder for median beat ECGs to get explainable factors. These factors can subsequently be used in other statistical models to perform diagnosis or prediction.

Next to these three architectures, optimized for deep learning on ECGs, the package contains many utilities and metrics to optimize training on ECGs. In the future, next to VAE-based techniques for explainability, other methods will be added too.

## FactorECG
The ECG can be compressed into 21 significant factors, that allow for reconstruction of the full median beat ECG. To encode the ECG into these factors, an unsupervised beta variational auto-encoder (VAE) was trained to reconstruct 1,144,331 median beat ECGs from these 21 factors. The current trained model is able to encode any median beat ECG into these 21 factors, and is able to interactively generate new ECGs by adjusting the factors.

The obtained ECG factors can be used to perform explainable diagnosis or prediction using common statistical methods. In the paper, we show that only using the 21 interpretable factors, we are able to reach performances similar to a state-of-the-art ‘black box’ DNN in standard ECG interpretation, and novel ECG use cases, such as diagnosis of reduced ejection fraction and prediction of one-year mortality.

An interactive tool to visualize the FactorECG can be found at https://decoder.ecgx.ai, while a tool to transform your ECGs into FactorECGs is available at https://encoder.ecgx.ai.

## How to install   
```bash
# via PyPI
pip install ecgxai

# or from source
git clone https://github.com/rutgervandeleur/ecgxai.git
pip install .
```

## Examples
### Supervised training
Please find some examples to use our package to train supervised models on 12-lead ECGs in the `examples/classification` folder.

### Training of the FactorECG VAE
Please find some examples to use our package to train the VAE as proposed in the [FactorECG](https://www.medrxiv.org/content/10.1101/2022.01.04.22268759v1) paper in the `examples/vae` folder.


## Citations 
Don't forget to cite one of our papers when using the package to train on your data!

#### CausalCNN architecture
```
@article{10.22489/cinc.2020.253, 
year = {2020}, 
title = {{Automated Comprehensive Interpretation of 12-lead Electrocardiograms Using Pre-trained Exponentially Dilated Causal Convolutional Neural Networks}}, 
author = {Bos, Max N and Leur, Rutger R van de and Vranken, Jeroen F and Gupta, Deepak K and Harst, Pim van der and Doevendans, Pieter A and Es, René van}, 
journal = {2020 Computing in Cardiology}, 
issn = {2325-887X}, 
doi = {10.22489/cinc.2020.253}, 
abstract = {{Correct interpretation of the electrocardiogram (ECG) is critical for the diagnosis of many cardiac diseases, and current computerized algorithms are not accurate enough to provide automated comprehensive interpretation of the ECG. This study aimed to develop and validate the use of a pre-trained exponentially dilated causal convolutional neural network for interpretation of the ECG as part of the 2020 Physionet/Computing in Cardiology Challenge. The network was pre-trained on a physician-annotated dataset of 254,044 12-lead ECGs. The weights of the pre-trained network were partially frozen, and the others were finetuned on the challenge dataset of 42,511 ECGs. 10-fold cross-validation was applied and the best performing model in each fold was selected and used to construct an ensemble. The proposed method yielded a cross-validated area under the receiver operating curve (AU-ROC) of 0.939 ± 0.004 and a challenge score of 0.565 ± 0.005. Evaluation on the hidden test set resulted in a score of 0.417, placing us 7th out of 41 in the official ranking (team name UMCUVA). We demonstrated that an ensemble of exponentially dilated causal convolutional networks and pre-training on a large dataset of ECGs from a different country and device manufacturer performs excellent for interpretation of ECGs.}}, 
pages = {1--4}, 
volume = {00}, 
keywords = {}
}
```

#### Double residual architecture and qLST
```
@article{undefined, 
year = {2021}, 
title = {{Interpretable ECG classification via a query-based latent space traversal (qLST)}}, 
author = {Vessies, Melle B and Vadgama, Sharvaree P and Leur, Rutger R van de and Doevendans, Pieter A and Hassink, Rutger J and Bekkers, Erik and Es, René van}, 
journal = {arXiv}, 
eprint = {2111.07386}, 
abstract = {{Electrocardiography (ECG) is an effective and non-invasive diagnostic tool that measures the electrical activity of the heart. Interpretation of ECG signals to detect various abnormalities is a challenging task that requires expertise. Recently, the use of deep neural networks for ECG classification to aid medical practitioners has become popular, but their black box nature hampers clinical implementation. Several saliency-based interpretability techniques have been proposed, but they only indicate the location of important features and not the actual features. We present a novel interpretability technique called qLST, a query-based latent space traversal technique that is able to provide explanations for any ECG classification model. With qLST, we train a neural network that learns to traverse in the latent space of a variational autoencoder trained on a large university hospital dataset with over 800,000 ECGs annotated for 28 diseases. We demonstrate through experiments that we can explain different black box classifiers by generating ECGs through these traversals.}}, 
keywords = {}
}
```

#### FactorECG and the explainable pipeline for interpretation of ECGs
```
@article{van de Leur2022.01.04.22268759, 
year = {2022}, 
title = {{Inherently explainable deep neural network-based interpretation of electrocardiograms using variational auto-encoders}}, 
author = {Leur, Rutger R. van de and Bos, Max N. and Taha, Karim and Sammani, Arjan and Duijvenboden, Stefan van and Lambiase, Pier D. and Hassink, Rutger J. and Harst, Pim van der and Doevendans, Pieter A. and Gupta, Deepak K. and Es, René van}, 
journal = {medRxiv}, 
doi = {10.1101/2022.01.04.22268759}, 
eprint = {https://www.medrxiv.org/content/early/2022/01/05/2022.01.04.22268759.full.pdf}, 
url = {https://www.medrxiv.org/content/early/2022/01/05/2022.01.04.22268759}, 
keywords = {}
}
```   