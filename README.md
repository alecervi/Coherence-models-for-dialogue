# Coherence-models-for-dialogue
This is the repository for the Interspeech 2018 paper ["Coherence models for dialogue"](https://arxiv.org/pdf/1806.08044.pdf) .
If you use our code, please cite our paper:

Cervone, A., Stepanov, E.A., & Riccardi, G. (2018). Coherence Models for Dialogue. Interspeech.

## Prerequisites

- python 2.7+
- spacy version 1
- tqdm
- mlxtend

The code generates input feature vectors files for [SVM light](http://svmlight.joachims.org/) version 6.02, so in order to finally train the models you will need to install that as well.


## Data preprocessing

The data used in the experiments (i.e. only the grid files, since source corpora are under licenses) is available in the `data` folder. Furthermore, the scripts we used to generate the data from source corpora is available. See the README file in the `data/` folder for further details.
The corpora preprocessing step in the `corpus` folder is a modification of the scripts from [this library](https://github.com/ColingPaper2018/DialogueAct-Tagger).

### Where do I find the source corpora used in the experiments?
- BT Oasis is available [via email request](http://groups.inf.ed.ac.uk/oasis/)
- The Switchboard Dialogue Act Corpus is available [here](https://web.stanford.edu/~jurafsky/swb1_dialogact_annot.tar.gz)
- AMI is available for download from [this page](http://groups.inf.ed.ac.uk/ami/download/)


## Getting started

### Generate features vectors with the provided data

After having unzipped the file `data/data.zip` (the data used in our experiments), you can directly generate the feature vectors for SVM light for the corpus Oasis with default parameters:
```
python train_models.py -g Oasis
```
You can find the generated feature vector files in the newly created path `experiments/Oasis/reordering/egrid_-coref/` , divided according to the train/dev/test splits.

### Train

Then, to train a model using SVM light you can run:
```
svm_learn -z p experiments/Oasis/reordering/egrid_-coref/Oasis_sal1_range2_2_train.dat my_model
```

### Predict

To classify the test set using your newly trained model:
```
svm_classify experiments/Oasis/reordering/egrid_-coref/Oasis_sal1_range2_2_test.dat my_model my_prediction
```

### Evaluate

Finally, to get the accuracy and other metrics reported in our paper for the newly trained model on the testset:
```
python eval_svmlight_output.py --testfile experiments/Oasis/reordering/egrid_-coref/Oasis_sal1_range2_2_test.dat --predfile my_prediction
```
If you don't specify the file with the model's predictions (```--predfile```), the script reports the performance of a random baseline on the testset using the same metrics:
```
python eval_svmlight_output.py --testfile experiments/Oasis/reordering/egrid_-coref/Oasis_sal1_range2_2_test.dat
```
### Data generation from the corpus

For generating grids for training entity grid models using only entities (without coreference) for the corpus Oasis (provided you gave the correct path to the Oasis source files) in verbose mode:
```
python generate_grid.py Oasis egrid_-coref egrid_-coref data/ -v
```
After having generated the original grids, generate shuffled grids (for the reordering task) for the same corpus:
```
python generate_shuffled.py -gs Oasis
```


## Link to our experiments files
You can get all our experiments files (obtained with the above procedure) [here](https://www.dropbox.com/s/1sewemx965o2jec/experiments.zip?dl=0)
