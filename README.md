# machine_translation_experiments
Machine translation experiments

## Tranfomers through fairseq
  Fairseq(-py) is a sequence modeling toolkit from Facebook AI Research that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks. 
  URL: https://github.com/pytorch/fairseq

### Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

Please follow the instructions here to install PyTorch: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq with `pip`:
```
pip install fairseq
```

### Model 
  Description | Dataset | Model | Test set(s)
  ---|---|---|---
  Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

### Results 
  ```
  $bash fairseq_wmt18_tranformer.sh
  BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt18+tok.13a+version.1.3.1 = 39.0 67.7/44.7/32.2/23.8 (BP = 1.000 ratio = 1.040 hyp_len = 66820 ref_len = 64276)
  Elapsed time (secs): 322
  ```

  The model achieves 39 BLEU and it needs 322 seconds for predicting on the test of WMT18 (2998 sentences).  
  

