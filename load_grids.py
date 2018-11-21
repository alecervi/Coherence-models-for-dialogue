from __future__ import print_function
from numpy import cumsum
# from generate_grid import dump_csv
import argparse
import logging
import os
import pandas as pd


class GridLoader(object):

    ''' Class to load grids (together with set-up params) '''

    def __init__(self, grid_folder=None):
        try:
            assert os.path.exists(grid_folder)  # folder exists
            assert os.path.exists(grid_folder + "Params.csv")
        except AssertionError:
            print("The folder " + grid_folder + " does not exist.")
            exit(1)
        try:
            assert os.path.exists(grid_folder + "Params.csv")
        except AssertionError:
            print("Params file missing in folder " + grid_folder)
            exit(1)
        self.grid_folder = grid_folder
        self.grids = []
        self.params = []

    def reformat_params(self, params):
        # params_tups = zip(params[params.columns[0]], params[params.columns[1]])
        params.columns = params.iloc[0]
        params.drop(params.index[0], inplace=True)
        # params.reindex(params.index.drop(1))
        return params

    def load_data(self, df=True):
        files = [f for f in os.listdir(self.grid_folder)
                 if os.path.isfile(os.path.join(self.grid_folder, f))]
        grids = {f.split('.csv')[0]: pd.read_csv(os.path.join(self.grid_folder, f), engine="c") for f in files
                 if f not in ['Params.csv', '.DS_Store.csv']}
        params = pd.read_csv(os.path.join(self.grid_folder, 'Params.csv'), header=None, engine="c").T
        params = self.reformat_params(params)
        # params = grids['Params'].T
        # grids.pop('Params')
        self.grids = grids
        self.params = params
        return grids, params

    def get_data(self):
        return self.grids, self.params

    def percentage_split(self, seq, percentages):
        # TODO: acknowledge
        percentages = list(percentages)
        cdf = cumsum(percentages)
        assert cdf[-1] == 1.0
        stops = list(map(int, cdf * len(seq)))
        return [seq[a:b] for a, b in zip([0] + stops, stops)]

    def split_dataset(self, dataset, split_percentages=(0.2, 0.8)):
        # shuffle(dataset)
        print("All dataset size: ", len(dataset))
        test, training_all = self.percentage_split(dataset, split_percentages)
        print("Test size: ", len(test))
        print("All training data size: ", len(training_all))
        validation, training = self.percentage_split(training_all, split_percentages)
        print("Strict training size: ", len(training))
        print("Validation size: ", len(validation))
        return training, validation, test

    def get_training_test_splits(self, data_folder='data/',
                                 corpus_name = 'Switchboard',
                                 default_grids_folder='egrid_-coref/',
                                 split_filename ="Train_Validation_Test_split"):

        # It returns training/validation/test splits if available,
        # otherwise it generates new splits

        splits = None
        grids_folder = data_folder + corpus_name + '/' + default_grids_folder
        try:
            assert os.path.exists(data_folder + corpus_name + '/' + split_filename + ".csv")
            print('Using already available train dev test split')
            splits = pd.read_csv(data_folder + corpus_name + '/' + split_filename + ".csv", dtype=str, keep_default_na=False)
        except AssertionError:
            print('No training/val/test splits already available')
            try:
                assert os.path.exists(grids_folder)
                dataset = [x.rstrip('.csv') for x in os.listdir(grids_folder) if x not in ['Params.csv', '.DS_Store.csv']]
                training, validation, test = self.split_dataset(dataset)

                splits = pd.DataFrame(dict([(k, pd.Series(v))
                                            for k, v in [('training', training),
                                                         ('validation', validation),
                                                         ('test', test)]]))
                # splits = pd.DataFrame(dict([(k, v)
                #                             for k, v in [('training', training),
                #                                          ('validation', validation),
                #                                          ('test', test)]]))

                splits.to_csv(path_or_buf=data_folder + corpus_name + '/' + split_filename + '.csv')
                # print('Test: ', test)
                # print('Check test split : ', dataset.index(test[-1]))
                # print('Check val split : ', dataset.index(validation[-1]))
                # print('Check train split : ', dataset.index(training[-1]))

            except AssertionError:
                print("Missing data in folder " + grids_folder)
                exit(1)

        return splits

def main():
    grids_path = 'data/egrid_-coref/'
    grid_loader = GridLoader(grids_path)
    # print(grid_loader.load_data())
    print('Splits: ', grid_loader.get_training_test_splits(corpus_name='Oasis').shape)



if __name__ == '__main__':
    main()




