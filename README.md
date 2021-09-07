# Iconary
This is the code for our paper
"Iconary: A Pictionary-Based Game for Testing MultimodalCommunication with Drawings and Text"

## Install
Install python >= 3.6 and pytorch >= 1.6.0. This project has been tested with torch==1.7.1, but 
later versions might work.

Then install the extra requirements:

`pip install -r requirements`

Finally add the top-level directory to PYTHONPATH:
```
cd iconary
export PYTHONPATH=`pwd`
```

## Data
Datasets will be downloaded and cached automatically as needed, `file_paths.py`
shows where the files will be stored. By defaults, datasets are stored in ~/data/iconary.

If you want to download the data manually, the dataest can be downloaded here:

- [Train](https://ai2-vision-iconary.s3.amazonaws.com/public-datasets/train.json)
- [IND Val](https://ai2-vision-iconary.s3.amazonaws.com/public-datasets/ind-valid.json)
- [IND Test](https://ai2-vision-iconary.s3.amazonaws.com/public-datasets/ind-test.json)
- [OOD Val](https://ai2-vision-iconary.s3.amazonaws.com/public-datasets/ood-valid.json)
- [OOD Test](https://ai2-vision-iconary.s3.amazonaws.com/public-datasets/ood-test.json)

We release the complete datasets without held-out labels since computing the automatic metrics for 
both the Guesser and Drawer requires the entire game to be known. Models should only be trained on the train set
and researchers should avoid looking/evaluating on the test sets as much as possible.

## Models
We release the following models on S3:

Guesser:
- TGuesser: s3://ai2-vision-iconary/public-models/tguesser-3b/
- w/T5-Large: s3://ai2-vision-iconary/public-models/tguesser-large/
- w/T5-Base: s3://ai2-vision-iconary/public-models/tguesser-base/

Drawer:
- TDrawer: s3://ai2-vision-iconary/public-models/tdrawer-large/
- w/T5-Base: s3://ai2-vision-iconary/public-models/tdrawer-base/


To use these models, download the entire directory. For example:

```
mkdir -p models
aws s3 cp --recursive s3://ai2-vision-iconary/public-models/tguesser-base models/tguesser-base
```

## Train
### Guesser

Train TGuesser with:

`python iconary/experiments/train_guesser.py --pretrained_model t5-base --output_dir models/tguesser-base`

Note our full model use `--pretrained_model t5-b3`, but that requries a >16GB RAM GPU to run.

### Drawing

Train TDrawer with:

`python iconary/experiments/train_drawer.py --pretrained_model t5-base --output_dir models/tdrawer-base --grad_accumulation 2`

Note our full model use `--pretrained_model t5-large`, but that requires a >16GB RAM GPU to run.


## Automatic Evaluation
These scripts generate drawings/guesses for games in human/human games, and computes 
automatic metrics from those drawings/guesses.
Note our generation scripts will use all GPUs that they can find with `torch.cuda.device_count()`, to control where
it runs use the `CUDA_VISIBLE_DEVICES` environment variable.

### Guesser
To compute automatic metrics for the Guesser, first generate guesses as:

`python iconary/experiments/generate_guesses.py path/to/model --dataset ood-valid --output_file guesses.json --unk_boost 2.0`

Note that most of our evaluations are done using `--unk_boost 2.0` which implements rare-word boosting.

This script will report our automatic metrics, but they can also be re-computed using:

`python iconary/experiments/eval_guesses.py guesses.json`

### Drawer
Generate drawings with:

`python iconary/experiments/generate_drawings.py path/to/model --dataset ood-valid --output_file drawings.json`

This script will report our automatic metrics, but they can also be re-computed using:

`python iconary/experiments/eval_drawings.py drawings.json`


## Human/AI Evaluation
Our code for running human/AI games is not currently released, if you are interested in running your own trials
contact us and we can help you follow our human/AI setup. 

## Cite
If you use this work, please cite:

"Iconary: A Pictionary-Based Game for Testing MultimodalCommunication with Drawings and Text". 
Christopher Clark, Jordi Salvador, Dustin Schwenk, Derrick Bonafilia, Mark Yatskar, Eric Kolve, 
Alvaro Herrasti, Jonghyun Choi, Sachin Mehta, Sam Skjonsberg, Carissa Schoenick, Aaron Sarnat, 
Hannaneh Hajishirzi, Aniruddha Kembhavi, Oren Etzioni, Ali Farhadi. In EMNLP 2021.

