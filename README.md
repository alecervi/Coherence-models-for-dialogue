# Coherence-models-for-dialogue
This is the repository for the Interspeech 2018 paper ["Coherence models for dialogue"](https://arxiv.org/pdf/1806.08044.pdf) .
If you use our code, please cite our paper:

Cervone, A., Stepanov, E.A., & Riccardi, G. (2018). Coherence Models for Dialogue. Interspeech.

## Prerequisites

- python 2.7+
- spacy version 1
- tqdm

## Data preprocessing

The data used in the experiments (i.e. only the grid files, since source corpora are under licences) is available in the `data` folder. Furthermore, the scripts we used to generate the data from source corpora is available. See the README file in the `data/` folder for further details.

## Getting started

Generate grids for training entity grid models using only entities (without coreference) for the corpus Oasis (provided you gave the correct path to the Oasis source files) in verbose mode:
```
python generate_grid.py Oasis egrid_-coref egrid_-coref data/ -v
```

## Coming soon

Code for experiments is coming soon!
