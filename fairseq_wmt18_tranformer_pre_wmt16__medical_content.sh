#!/usr/bin/env bash
# First install sacrebleu and sentencepiece
pip install sacrebleu sentencepiece

##############
SRC=en
DEST=de
CORPUS=wmt18
MODEL=wmt16.en-de.joined-dict.transformer
CODE=code_wmt18

mkdir -p data-bin
if [ ! -d "data-bin/${MODEL}" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    curl https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2 | tar xvjf - -C data-bin
else
    echo ">>> looks like you have already downloaded the model ..."
fi
#curl https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2 | tar xvjf - -C data-bin
#curl https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.bz2 | tar xvjf - -C data-bin

########### wmt18
if [ ! -d "mosesdecoder" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    echo 'Cloning Moses github repository (for tokenization scripts)...'
    git clone https://github.com/moses-smt/mosesdecoder.git
else
    echo ">>> looks like you have already downloaded  https://github.com/moses-smt/mosesdecoder.git  ..."
fi

if [ ! -d "subword-nmt" ]; then
    # Control will enter here if $DIRECTORY doesn't exist.
    echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
    git clone https://github.com/rsennrich/subword-nmt.git
else
    echo ">>> looks like you have already downloaded  https://github.com/rsennrich/subword-nmt.git  ..."
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt

START_TIME=$SECONDS
cat medical_content.txt | $NORM_PUNC -l en | $TOKENIZER -a -l en -q | python $BPEROOT/apply_bpe.py -c ${CODE} | fairseq-interactive data-bin/${MODEL} --path data-bin/${MODEL}/model.pt --remove-bpe --buffer-size 1024 --batch-size 16 -s ${SRC} -t ${DEST} | grep -P "^H" |cut -f 3- | $DETOKENIZER -l ${DEST} -q > medical_content.txtde.out

cat ${CORPUS}.test.${SRC}-${DEST}.${SRC}.out | sacrebleu -t ${CORPUS} -l ${SRC}-${DEST} -lc

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Elapsed time (secs): $ELAPSED_TIME"
