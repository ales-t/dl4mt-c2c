Fully Character-Level Neural Machine Translation (forked from Cho's repository)
==================================

How to run:

build the character dictionary:

TODO

run training:

python \
  /mounts/data/proj/himl/tamchyna/fork-dl4mt-c2c/char2char/train_bi_char2char.py \
  -data_path=`pwd`/data \
  -model_path=`pwd`/model \
  -source_lang SRC_SUFFIX \
  -target_lang TGT_SUFFIX \
  -train_corpus=train.tc \
  -dev_corpus=dev.tc \
  -dict=char-dict

run translation:

python /mounts/data/proj/himl/tamchyna/fork-dl4mt-c2c/translate/translate_char2char.py \
  -input=data/test.tc.en \
  -source_dict=data/char-dict.en \
  -target_dict=data/char-dict.cs \
  -model=modelbi-char2char.npz \
  > output.txt
