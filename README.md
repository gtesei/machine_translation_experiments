# machine_translation_experiments
Machine translation experiments

## Requirements and Installation

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

Fairseq(-py) is a sequence modeling toolkit from Facebook AI Research that allows researchers and developers to train custom models for translation, summarization, language modeling and other text generation tasks.

URL: https://github.com/pytorch/fairseq

## [Tranfomers - Ott et al., 2018](https://arxiv.org/abs/1806.00187)

### Model 
  Description | Dataset | Model | Test set(s)
  ---|---|---|---
  Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)

### Results 
  ```
  $bash fairseq_wmt18_tranformer_pre_wmt16.sh
  ...
  BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt18+tok.13a+version.1.3.1 = 39.0 67.7/44.7/32.2/23.8 (BP = 1.000 ratio = 1.040 hyp_len = 66820 ref_len = 64276)
  Elapsed time (secs): 292
  ```

  The model achieves ~39 BLEU score and it needs ~292 seconds for predicting on the test of WMT18 (2,998 sentences). The state of the art (4/10/2019) for WMT18 ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381); WMT'18 winner) claims 46.53 BLEU score although it is much slower.   

###  Generating Translations Interactively

   ```
   > MODEL_DIR=data-bin/wmt16.en-de.joined-dict.transformer
   > fairseq-interactive --path $MODEL_DIR/model.pt $MODEL_DIR --beam 5 --source-lang en --target-lang de
   Namespace(beam=5, buffer_size=1, cpu=False, data=['data-bin/wmt16.en-de.joined-dict.transformer'], diverse_beam_groups=-1, diverse_beam_strength=0.5, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', input='-', lazy_load=False, left_pad_source='True', left_pad_target='False', lenpen=1, log_format=None, log_interval=1000, match_source_len=False, max_len_a=0, max_len_b=200, max_sentences=1, max_source_positions=1024, max_target_positions=1024, max_tokens=None, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', nbest=1, no_beamable_mm=False, no_early_stop=False, no_progress_bar=False, no_repeat_ngram_size=0, num_shards=1, num_workers=0, path='data-bin/wmt16.en-de.joined-dict.transformer/model.pt', prefix_size=0, print_alignment=False, quiet=False, raw_text=False, remove_bpe=None, replace_unk=None, required_batch_size_multiple=8, results_path=None, sacrebleu=False, sampling=False, sampling_temperature=1, sampling_topk=-1, score_reference=False, seed=1, shard_id=0, skip_invalid_size_inputs_valid_test=False, source_lang='en', target_lang='de', task='translation', tensorboard_logdir='', threshold_loss_scale=None, unkpen=0, unnormalized=False, upsample_primary=1, user_dir=None)
   | [en] dictionary: 32768 types
   | [de] dictionary: 32768 types
   | loading model(s) from data-bin/wmt16.en-de.joined-dict.transformer/model.pt
   | Type the input sentence and press return:
   It is annoying when geographical maps are not up-to-date.
   S-0     It is <unk> when geographical maps are not <unk>
   H-0     -0.498351514339447      Es ist kostenlos , wenn geograph@@ ische Karten nicht heruntergeladen werden
   P-0     -1.0888 -0.1411 -0.3799 -0.1798 -0.1026 -1.4384 -0.0999 -0.0966 -0.2404 -1.4965 -0.2622 -0.4542
   Then it becomes clear how the towns and municipalities in the distribution area of Munich's Merkur newspaper have changed since the 19th century.
   S-1     Then it becomes clear how the towns and <unk> in the distribution area of <unk> <unk> newspaper have changed since the 19th <unk>
   H-1     -0.41815847158432007    Dann wird deutlich , wie sich die Städte und <unk> im Distri@@ bu@@ tions@@ gebiet der <unk> jährigen Zeitung seit dem 19 . <unk> verändert haben .
   P-1     -0.3165 -0.3109 -0.6439 -0.1497 -0.1608 -0.2241 -0.4928 -0.1953 -0.1754 -2.1586 -0.2352 -1.3033 -0.0397 -0.0334 -0.6105 -0.3980 -0.2919 -0.6659 -0.1218 -0.1431 -0.1675 -0.1142 -0.1018 -1.8401 -0.3494 -0.1096 -0.2357 -0.1196	   
   ```
###  Why it is necessary tokenizing/detokenizing with so many perl scripts

   ```
   > MODEL_DIR=data-bin/wmt16.en-de.joined-dict.transformer
   > fairseq-interactive --path $MODEL_DIR/model.pt $MODEL_DIR --beam 5 --source-lang en --target-lang de
   Namespace(beam=5, buffer_size=1, cpu=False, data=['data-bin/wmt16.en-de.joined-dict.transformer'], diverse_beam_groups=-1, diverse_beam_strength=0.5, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', input='-', lazy_load=False, left_pad_source='True', left_pad_target='False', lenpen=1, log_format=None, log_interval=1000, match_source_len=False, max_len_a=0, max_len_b=200, max_sentences=1, max_source_positions=1024, max_target_positions=1024, max_tokens=None, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', nbest=1, no_beamable_mm=False, no_early_stop=False, no_progress_bar=False, no_repeat_ngram_size=0, num_shards=1, num_workers=0, path='data-bin/wmt16.en-de.joined-dict.transformer/model.pt', prefix_size=0, print_alignment=False, quiet=False, raw_text=False, remove_bpe=None, replace_unk=None, required_batch_size_multiple=8, results_path=None, sacrebleu=False, sampling=False, sampling_temperature=1, sampling_topk=-1, score_reference=False, seed=1, shard_id=0, skip_invalid_size_inputs_valid_test=False, source_lang='en', target_lang='de', task='translation', tensorboard_logdir='', threshold_loss_scale=None, unkpen=0, unnormalized=False, upsample_primary=1, user_dir=None)
   | [en] dictionary: 32768 types
   | [de] dictionary: 32768 types
   | loading model(s) from data-bin/wmt16.en-de.joined-dict.transformer/model.pt
   | Type the input sentence and press return:
   It is annoying when geographical maps are not up-to-date.
   S-0     It is <unk> when geographical maps are not <unk>
   H-0     -0.498351514339447      Es ist kostenlos , wenn geograph@@ ische Karten nicht heruntergeladen werden
   P-0     -1.0888 -0.1411 -0.3799 -0.1798 -0.1026 -1.4384 -0.0999 -0.0966 -0.2404 -1.4965 -0.2622 -0.4542
   It is anno@@ ying when geographical maps are not up @-@ to @-@ date .
   S-2     It is anno@@ ying when geographical maps are not up @-@ to @-@ date .
   H-2     -0.19444942474365234    Es ist är@@ gerlich , wenn geografische Karten nicht auf dem neuesten Stand sind .
   P-2     -0.3297 -0.2593 -0.1436 -0.0003 -0.0079 -0.1646 -0.3046 -0.2832 -0.1375 -0.9979 -0.0535 -0.4037 -0.0023 -0.0190 -0.0040 -0.0002
   ```

## [Convolutional - Gehring et al., 2017](https://arxiv.org/abs/1705.03122)

### Model 
  Description | Dataset | Model | Test set(s)
  ---|---|---|---
  Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2)

### Results 
  ```
  $bash fairseq_wmt18_conv_pre_wmt14.sh
  ...
  BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt18+tok.13a+version.1.3.1 = 37.1 66.5/42.8/30.3/22.0 (BP = 1.000 ratio = 1.021 hyp_len = 65616 ref_len = 64276)
  Elapsed time (secs): 221
  ```

  The model achieves ~37 BLEU score and it needs ~221 seconds for predicting on the test of WMT18 (2,998 sentences). The state of the art (4/10/2019) for WMT18 ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381); WMT'18 winner) claims 46.53 BLEU score although it is much slower.


###  Generating Translations Interactively

   ```
   > MODEL_DIR=data-bin/wmt16.en-de.joined-dict.transformer
   > fairseq-interactive --path $MODEL_DIR/model.pt $MODEL_DIR --beam 5 --source-lang en --target-lang de]
   Namespace(beam=5, buffer_size=1, cpu=False, data=['data-bin/wmt14.en-de.fconv-py'], diverse_beam_groups=-1, diverse_beam_strength=0.5, fp16=False, fp16_init_scale=128, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', input='-', lazy_load=False, left_pad_source='True', left_pad_target='False', lenpen=1, log_format=None, log_interval=1000, match_source_len=False, max_len_a=0, max_len_b=200, max_sentences=1, max_source_positions=1024, max_target_positions=1024, max_tokens=None, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, model_overrides='{}', nbest=1, no_beamable_mm=False, no_early_stop=False, no_progress_bar=False, no_repeat_ngram_size=0, num_shards=1, num_workers=0, path='data-bin/wmt14.en-de.fconv-py/model.pt', prefix_size=0, print_alignment=False, quiet=False, raw_text=False, remove_bpe=None, replace_unk=None, required_batch_size_multiple=8, results_path=None, sacrebleu=False, sampling=False, sampling_temperature=1, sampling_topk=-1, score_reference=False, seed=1, shard_id=0, skip_invalid_size_inputs_valid_test=False, source_lang='en', target_lang='de', task='translation', tensorboard_logdir='', threshold_loss_scale=None, unkpen=0, unnormalized=False, upsample_primary=1, user_dir=None)
   | [en] dictionary: 40358 types
   | [de] dictionary: 42714 types
   | loading model(s) from data-bin/wmt14.en-de.fconv-py/model.pt
   | Type the input sentence and press return:
   It is annoying when geographical maps are not up-to-date.
   S-0     It is <unk> when geographical maps are not <unk>
   H-0     -0.9639539122581482     Es ist zu spät , wenn geografische Karten nicht .
   P-0     -0.5108 -0.1752 -2.6613 -3.2290 -0.1386 -0.0198 -0.3703 -0.1186 -0.0827 -3.2972 -0.0000
   Then it becomes clear how the towns and municipalities in the distribution area of Munich's Merkur newspaper have changed since the 19th century.
   S-1     Then it becomes clear how the towns and municipalities in the distribution area of <unk> <unk> newspaper have changed since the 19th <unk>
   H-1     -0.6274912357330322     Dann wird deutlich , wie sich die Städte und Gemeinden im Vertriebs@@ gebiet der Zeitung &quot; Die Stadt &quot; seit dem 19. Jahrhundert verändert haben .
   P-1     -0.4565 -0.4069 -0.6206 -0.0102 -0.1406 -0.0521 -0.1865 -0.2732 -0.1224 -0.0755 -0.3305 -1.7258 -0.5431 -0.1740 -2.5943 -1.3362 -3.7855 -2.1625 -0.2224 -0.0342 -0.0943 -0.1767 -0.9929 -0.4092 -0.0092 -0.0060 -0.0009
   ```

### Model 
  Description | Dataset | Model | Test set(s)
  ---|---|---|---
  Transformers ([Otto et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)

### Results 
  ```
  $bash fairseq_wmt14_tranformer_en_fr.sh
  ...
  BLEU+case.lc+lang.en-fr+numrefs.1+smooth.exp+test.wmt14+tok.13a+version.1.3.1 = 37.1 64.8/43.0/30.6/22.1 (BP = 1.000 ratio = 1.008 hyp_len = 77956 ref_len = 77306)
  Elapsed time (secs): 179
  ```



### Fairseq Support 
  Language | Pre-trained model from fairseq | link | Notes
  ---|---|---|---
  zh | NO (4/11/2019) | -  | - 
  fr | YES  | https://github.com/pytorch/fairseq/tree/master/examples/translation  | Not sure fr-en (4/11/2019)
  de | YES  | https://github.com/pytorch/fairseq/tree/master/examples/translation  | Not sure de-en (4/11/2019)
  ja | NO (4/11/2019) | -  | - 
  pt | NO (4/11/2019) | -  | - 
  es | NO (4/11/2019) | -  | - 
  ru | NO (4/11/2019) | -  | - 
  tr | NO (4/11/2019) | -  | - 
  

### Comparison (BLEU score)
  Language | Pre-trained model fairseq | ST | State-of-the-art
  ---|---|---|---
  en-de (wmt18) | 39.0 | 35.2  | 46.53 
  en-fr (wmt14)| 37.1  | 35.5  | 43.2
  


