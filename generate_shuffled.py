from __future__ import print_function
from future.utils import iteritems
from builtins import dict
from corpus.Switchboard.Switchboard import Switchboard
from generate_grid import dump_csv, GridGenerator
from load_grids import GridLoader
from itertools import groupby
from generate_grid import corpora_paths, get_corpus
from pandas.util.testing import assert_frame_equal
from operator import itemgetter
import itertools
import argparse
import pandas as pd
import warnings
import copy
import tqdm
import re
import logging
import random
import os
import numpy as np

warnings.simplefilter('error', UserWarning)


class GridShuffler(object):
    '''Class to generate and save shuffled files'''

    def __init__(self, grid_folder=None, grid_loader=None, grid_generator=None):

        if grid_loader:
            self.grid_loader = grid_loader
        else:
            try:
                assert os.path.exists(grid_folder)
            except AssertionError:
                print("The folder " + grid_folder + " does not exist.")
                exit(1)
            self.grid_loader = GridLoader(grid_folder)

        self.grids, self.grids_params = self.grid_loader.get_data()
        if not self.grids:
            self.grids, self.grids_params = self.grid_loader.load_data()

        if grid_generator:
            self.grid_generator = grid_generator
        else:
            self.grid_generator = GridGenerator(coref='no_coref', nlp='no_nlp')

        self.turn2da = {} # tuples dct
        self.da2turn = {}

    def update_grids_dct(self, grid_names):
        self.grids = {k:v for k,v in iteritems(self.grids) if k in grid_names}

    def get_intra_shuffle(self, len_dialogue, shuffles_number=20):
        index_shuff = range(len_dialogue)
        # print('Dial len: ', len(index_shuff))

        shuffled_orders = []
        # for i in range(shuffles_number):
        #     shuffled_order = random.sample(index_shuff, len(index_shuff))
        #     shuffled_orders.append(shuffled_order)
        shuff_i = 0
        # print('Shuffled orders: ', shuffled_orders)
        while shuff_i < shuffles_number:
            s = random.sample(index_shuff, len(index_shuff))
            if s != index_shuff:
                shuffled_orders.append(s)
                shuff_i += 1
            else:
                pass
        return shuffled_orders

    def generate_turn_shuffle_index(self, corpus, shuffles_number=20, min_dial_len =4, corpus_path=''):

        shuffled_dialogues = {}
        random.seed(0)

        for dialogue_id, dialogue in tqdm.tqdm(iteritems(corpus), total=len(corpus)):
            dialogue = self.grid_generator.group_turns(dialogue)
            # print('Dialogue: ', dialogue_id)

            # dialogue = dialogue[:8] # To delete
            len_dialogue = len(dialogue)

            # Minimum 5 turns and check we have a corresponding grid file
            if len_dialogue > min_dial_len and os.path.exists(corpus_path+dialogue_id+'.csv'):
                shuffled_orders = self.get_intra_shuffle(len_dialogue, shuffles_number=shuffles_number)
                max_ind = list(set([max(s) for s in shuffled_orders]))
                if len(max_ind)==1 and max_ind[0]==(len_dialogue-1) and max_ind[0]==list(set([len(s)-1 for s in shuffled_orders]))[0]:
                    shuffled_dialogues[dialogue_id] = shuffled_orders
                else:
                    print('ID: ', dialogue_id)
                    print('Dialogue orders messed up')
                    print('Len Dialogue: ', len_dialogue)
                    print('Dialogue check: ', [dialogue[i] for i in range(len_dialogue) if int(dialogue[i][-1]) != (i + 1)])
                    print('Shuff len :', [len(s) for s in shuffled_orders])
                    print('Shuff max :', max_ind)

        print('Len shuffled dialogues: ', len(shuffled_dialogues))
        return shuffled_dialogues

    def count_pairs(self):
        raise NotImplementedError

    def write_csv_indexes(self, shuffled_dialogues, folder_name='shuffled',
                          folder_path='data/', corpus_name='Switchboard', rewrite_new=False):
        full_path = folder_path + corpus_name + '/' + folder_name+'/'
        print('Out path: ', full_path)
        if not os.path.exists(full_path):
            if rewrite_new:
                os.makedirs(full_path)
                print('Shuffled indexes dir created: ', full_path)
            else:
                warnings.warn('The shuffled folder already exists and cannot be overwritten.')
        for dialogue_id, dialogue_indexes in iteritems(shuffled_dialogues):
            dump_csv(full_path+dialogue_id, dialogue_indexes)
        print('Shuffled files created')

    def create_shuffle_index_files(self, corpus_dct, corpus_name, shuffles_number=20, grids_path='', rewrite_new=False):
        shuffled_dialogues = self.generate_turn_shuffle_index(corpus_dct,
                                                              shuffles_number=shuffles_number,
                                                              corpus_path=grids_path)
        print('Len shuffled dialogues: ', len(shuffled_dialogues))
        self.write_csv_indexes(shuffled_dialogues,
                               corpus_name=corpus_name,
                               rewrite_new=rewrite_new)

    def check_match_shuff_original(self, shuff_path):
        # Check shuffled and original match
        if not os.path.exists(shuff_path):
            warnings.warn('The shuffled folder does not exist')
        shuffled_dialogues_list = [x.rstrip('.csv') for x in os.listdir(shuff_path)]
        is_match = all(grid_name in shuffled_dialogues_list for grid_name in self.grids)
        return is_match

    def get_turns_to_da_map(self, dialogue, grid_i_name, is_eot=False):
        group_maps = []
        previous_ind = 0
        # start/end tuples where index=turn
        for k, g in groupby(dialogue, itemgetter(3)):
            g = list(g)
            group_maps.append([previous_ind, (previous_ind)+len(g)])
            previous_ind += len(g)
        # print('2nd group: ', group_maps[1])
        # # Get das for turn-range
        # print('2nd group corresponding indexes: ', [dialogue[i] for i in range(group_maps[1][0], group_maps[1][-1])])
        # print('3d group: ', group_maps[2])
        # print('3d group corresponding indexes: ', [dialogue[i] for i in range(group_maps[2][0], group_maps[2][-1])])

        self.turn2da[grid_i_name] = {group_maps.index(g):g for g in group_maps}
        self.da2turn[grid_i_name] = {i:group_maps.index(g) for g in group_maps for i in range(g[0],g[-1])}
        return group_maps


    def map_index_shuffles_to_grids(self, permuted_indexes, grid, df=False):
        # Returns list of rows from original grid reordered according to permuted order
        # return [[grid.iloc[ind] for ind in perm] for perm in permuted_indexes]
        print('Permuted indexes: ', len(permuted_indexes))
        print('Grid shape: ', grid.shape)
        if df is False:
            perm_rows = [[grid.iloc[ind] for ind in perm] for perm in permuted_indexes]
        else:
            perm_rows = []
            # For permutation type
            for perm in permuted_indexes:
                perm_i_rows = [grid.iloc[ind] for ind in perm]
                perm_i_rows_df = pd.DataFrame.from_items([(c, [r[i] for r in perm_i_rows]) for i, c in enumerate(grid.columns)])
                perm_rows.append(perm_i_rows_df)

        return perm_rows

    def map_index_shuffles_to_grids_fast(self, permuted_indexes, grid):
        # Returns list of rows from original grid reordered according to permuted order
        # return [[grid.iloc[ind] for ind in perm] for perm in permuted_indexes]
        # print('Permuted indexes: ', len(permuted_indexes))
        # print('Grid shape: ', grid.shape)
        permuted_indexes = [permuted_indexes[0]]  # Testing
        perm_rows = []
        # For permutation type
        for perm in permuted_indexes:
            perm_grid = grid.copy()

            for ind_pos, ind in enumerate(perm):
                if ind < grid.shape[0] and ind_pos < grid.shape[0]:
                    perm_grid.iloc[ind_pos] = grid.iloc[ind].copy()
            perm_rows.append(perm_grid)

        return perm_rows

    def map_index_shuffles_to_grids_veryfast(self, permuted_indexes, grid):
        # Returns list of rows from original grid reordered according to permuted order
        # return [[grid.iloc[ind] for ind in perm] for perm in permuted_indexes]
        # print('Permuted indexes: ', len(permuted_indexes))
        # print('Grid shape: ', grid.shape)
        # permuted_indexes = [permuted_indexes[0]]  # Testing
        # perm_rows = []
        # # For permutation type
        # for perm in permuted_indexes:
        #     perm_rows.append(grid.reindex(perm).reset_index(drop=True))
        perm_rows = [grid.reindex(perm).reset_index(drop=True) for perm in permuted_indexes]
        return perm_rows

    def map_index_reinsertion_to_grids_veryfast(self, grid, times_number = 10):
        # Returns list of rows from original grid reordered according to permuted order
        perm_rows = []
        turns_number = grid.shape[0]

        np.random.seed(0)

        # print('Original grid: ', turns_number)
        # print('Original grid: ', grid)
        for i in range(min(turns_number, times_number)):

            sent_idx = np.random.randint(0, turns_number-1)
            # print('Turn index to reinsert: ', sent_idx)
            del_index = range(turns_number)
            del del_index[sent_idx]
            all_perm_turn_i = []
            for j in range(min(turns_number, times_number)):
                permuted_index = copy.deepcopy(del_index)
                cand = np.random.randint(0, turns_number)
                while cand == sent_idx:
                    cand = np.random.randint(0, turns_number)
                permuted_index.insert(cand, sent_idx)
                # print('Permuted index: ', permuted_index)
                all_perm_turn_i.append(grid.reindex(permuted_index).reset_index(drop=True))
                # print('Permuted grid: ', grid.reindex(permuted_index).reset_index(drop=True))
            perm_rows.append(all_perm_turn_i)

        return perm_rows

    def map_index_shuffles_to_grids_das(self, permuted_indexes, grid, group_maps, df=False):
        # Returns list of rows from original grid reordered according to permuted order
        perm_rows=[]
        # For permutation type
        for perm in permuted_indexes:
            perm_i_rows = []

            # For each turn index
            for ind in perm:
                # print('Perm ind: ', ind)
                # print('SO: ', group_maps[ind], ' Range: ', range(group_maps[ind][0], group_maps[ind][-1]))

                # Select all DA row indexes in that span
                for da_i in range(group_maps[ind][0]-1, group_maps[ind][-1]-1):
                    # print('Rows: ', grid.iloc[da_i])
                    if da_i < grid.shape[0]:
                        perm_i_rows.append(grid.iloc[da_i])
                    else:
                        pass
                        # print('DA index: ', da_i)
                        # print('Previous row selected: ', [r for r in grid.iloc[da_i-2]])

            if df is False:
                perm_i_rows.insert(0, grid.columns)
            else:
                # Convert into pandas DataFrame
                perm_i_rows = pd.DataFrame.from_items([(c, [r[i] for r in perm_i_rows]) for i, c in enumerate(grid.columns)])

            perm_rows.append(perm_i_rows)

        return perm_rows

    def map_index_shuffles_to_grids_das_fast(self, permuted_indexes, grid, group_maps, df=False):
        perm_rows = []
        # For permutation type

        permuted_indexes=[permuted_indexes[0]] # Testing
        for perm in permuted_indexes:
            # perm_i_rows = []
            perm_grid = grid.copy()
            ind_pos = 0

            # For each turn index
            for ind in perm:
                # print('Perm ind: ', ind, 'Ind pos: ', ind_pos)
                # print('SO: ', group_maps[ind], ' Range: ', range(group_maps[ind][0], group_maps[ind][-1]))

                # Select all DA row indexes in that span
                for da_i in range(group_maps[ind][0], group_maps[ind][-1]):
                    if ind_pos < grid.shape[0] and da_i < grid.shape[0]:
                        # print('Rows: ', grid.iloc[da_i])
                        # print('Rows: ', [(grid.columns[i], y) for i, y in enumerate(grid.iloc[da_i]) if y != '_'])
                        perm_grid.iloc[ind_pos] = grid.iloc[da_i].copy()
                        ind_pos += 1
                        # It cannot be only ind_pos but a range
                        # print('Perm Rows: ', [(perm_grid.columns[i], y) for i, y in enumerate(grid.iloc[da_i]) if y != '_'])
                        # perm_i_rows.append(grid.iloc[da_i])
                    else:
                        pass
                        # print('DA index: ', da_i)
                        # print('Previous row selected: ', [r for r in grid.iloc[da_i-2]])
            # print('Perm grid 3', [(perm_grid.columns[i], y) for i, y in enumerate(perm_grid.iloc[3]) if y != '_'])
            # print('Perm grid 1', [(perm_grid.columns[i], y) for i, y in enumerate(perm_grid.iloc[1]) if y != '_'])
            perm_rows.append(perm_grid)

        return perm_rows

    def map_index_shuffles_to_grids_das_veryfast(self, permuted_indexes, grid, group_maps, df=False):
        # permuted_indexes = [permuted_indexes[0]] # Testing
        perm_rows = [grid.reindex(self.turns_to_da(perm, group_maps)).reset_index(drop=True) for perm in permuted_indexes]
        # perm_rows = [grid.reindex(self.turns_to_da(perm, group_maps)) for perm in
        #              permuted_indexes] # Testing

        return perm_rows

    def map_index_reinsertion_to_grids_das_veryfast(self, turns_number, grid, group_maps, times_number=10, df=False):
        # permuted_indexes = [permuted_indexes[0]] # Testing
        np.random.seed(0)
        perm_rows = []
        turns_number = len(turns_number[0])
        for i in range(min(turns_number, times_number)):
            sent_idx = np.random.randint(0, turns_number-1)
            del_index = range(turns_number)
            del del_index[sent_idx]
            all_perm_turn_i = []
            for i in range(min(turns_number, times_number)):
                permuted_index = copy.deepcopy(del_index)
                cand = np.random.randint(0, turns_number)
                while cand == sent_idx:
                    cand = np.random.randint(0, turns_number)
                permuted_index.insert(cand, sent_idx)
                # print('Permuted index: ', permuted_index)
                # print('Len permuted index: ', len(permuted_index))
                # print('Group maps: ', group_maps)
                all_perm_turn_i.append(grid.reindex(self.turns_to_da(permuted_index, group_maps)).reset_index(drop=True))
            perm_rows.append(all_perm_turn_i)

        return perm_rows



    def turns_to_da(self, perm, group_maps):
        return list(itertools.chain(*map(lambda x: range(*group_maps[x]), perm)))

    def test_correspondance(self, y_row_ind, permut_i, permuted_indexes_i, dialogue_i, group_maps, grid_i, grid_i_name, permuted_files):

        print('Grid i shape: ', grid_i.shape)
        print('Dialogue len: ', len(dialogue_i))
        print('turn to DA:', self.turn2da[grid_i_name])
        print('DA to turn:', self.da2turn[grid_i_name])
        print('Len one permutation: ', len(permuted_files[grid_i_name][0]))
        print('Type one permutation: ', type(permuted_files[grid_i_name][0]))
        print('Shape one permutation: ', permuted_files[grid_i_name][0].shape)
        # print('Perm df: ', permuted_files[grid_i_name][0])
        print('Shape one permutation: ', permuted_files[grid_i_name][0].iloc[0])

        turn_y_permuted_index = permuted_indexes_i[permut_i][y_row_ind]  # ind to substitute: 44
        print('First permut, first row index: ', turn_y_permuted_index)
        print('Da group indexes corresponding to that turn index: ', group_maps[turn_y_permuted_index])
        print('Dialogue ref: ', [dialogue_i[i] for i in range(group_maps[turn_y_permuted_index][0],
                                                              group_maps[turn_y_permuted_index][-1])])

        # Check original grid rows corresponding to turn_y_permuted_index (44)
        print('Original grid da rows for turn ', turn_y_permuted_index)
        # For each DA index corresponding to the first turn in the permuted indexes
        for d in range(group_maps[turn_y_permuted_index][0],group_maps[turn_y_permuted_index][-1]):
            print('Original row name: ', grid_i.iloc[d].name)

        # Check permuted grid rows corresponding to y_row_ind (0)
        print('Permuted grid rows for turn ', y_row_ind)
        print('Start ', y_row_ind)
        print('End ', y_row_ind + (group_maps[turn_y_permuted_index][-1] - group_maps[turn_y_permuted_index][0]))

        # # For each DA index corresponding to the first turn in the permuted indexes
        # for d in range(y_row_ind,
        #                 y_row_ind+(group_maps[turn_y_permuted_index][-1] - group_maps[turn_y_permuted_index][0])):
        #     print('da index')
        #     print('Permuted grid first list Name: ', permuted_files[grid_i_name][permut_i][d].name)

        # for x in range(1, 100):
        #     current_perm = permuted_files[grid_i_name][permut_i]
        #     current_perm_indexes = permuted_indexes_i[permut_i]  # [48, 85, ]
        #     print('Permuted grid DA ind: ', x,' Orig DA ind: ', current_perm[x].name)
        #     print('Corresponding original Turn index: ',
        #           self.da2turn[grid_i_name][current_perm[x].name],
        #           ' Turn position in permuted: ', current_perm_indexes.index(self.da2turn[grid_i_name][current_perm[x].name]))
        #     print('Dialogue i ',
        #           dialogue_i[current_perm[x].name])

        # for x in range(0, 100):
        #     # print('Permuted grid DA ind: ', x, ' Permut row for that index: ', permuted_files[grid_i_name][permut_i].iloc[x])
        #     current_perm = permuted_files[grid_i_name][permut_i]
        #     current_perm_indexes = permuted_indexes_i[permut_i] # [48, 85, ]
        #     print('Permuted grid DA ind: ', x, ' Permut row for that index: ',
        #           [(current_perm.columns[i], y)
        #            for i, y in enumerate(current_perm.iloc[x]) if y != '_'])
        #     print('Corresponding original Turn index: ', permuted_indexes_i[permut_i][self.da2turn[grid_i_name][x]])

    def generate_shuffled_grids(self, folder_name='shuffled', corpus_name ='Switchboard',
                                folder_path='data/', corpus_dct=None,
                                only_grids = None, df=False, saliency=1, return_originals=False):
        # Check shuffled folder exist
        shuff_path = folder_path + corpus_name + '/' + folder_name+'/'
        print('Shuff path: ', shuff_path)
        self.check_match_shuff_original(shuff_path)
        self.grid_generator.corpus_stats(corpus_dct)

        permuted_files = {}
        if return_originals:
            original_files = {}

        grid_names = [x for x in self.grids if not re.match(r'.+\_s[0-9][0-9]*', x) and x!='Params']
        # print('Grid names: ', grid_names)

        if only_grids is not None:
            grid_names = [x for x in grid_names if x in only_grids and x!='.DS_Store']

        print('Len grids to permute: ', len(grid_names))
        self.update_grids_dct(grid_names)

        # Permute already generated grids according to shuffled indexes order
        for grid_i_name in tqdm.tqdm(grid_names):
            # print('Grid id: ', grid_i_name)
            grid_i = self.grids.get(grid_i_name)

            # Check saliency
            if saliency>1:
                grid_i.drop([col for col in grid_i if len([i for i in grid_i[col] if i != '_']) < saliency], axis=1)

            if return_originals:
                original_files[grid_i_name] = grid_i


            # print('Grid i columns: ', grid_i.columns)
            shuffled_indexes_i = pd.read_csv(shuff_path+grid_i_name+'.csv', header=None, engine="c").T

            # Get permutations
            permuted_indexes_i = [[ind for ind in shuffled_indexes_i[col]] for col in shuffled_indexes_i.columns]
            # print('First permut: ', permuted_indexes_i[0])
            # print('Groupby: ', self.grids_params.group_by[1])
            # print('End_of_turn_tag: ', self.grids_params.end_of_turn_tag[1])

            # Read Params of source grids: if 'group_by' : 'DAspan' (one more layer of mapping) or 'turns' (you can get it directly)
            if self.grids_params.group_by[1] != 'turns':
                dialogue_i = corpus_dct.get(grid_i_name)
                group_maps = self.get_turns_to_da_map(dialogue_i, grid_i_name)  # list of turns span
                # print('Group maps: ', group_maps)
                # permuted_files[grid_i_name] = self.map_index_shuffles_to_grids_das(permuted_indexes_i, grid_i, group_maps, df=df)
                permuted_files[grid_i_name] = self.map_index_shuffles_to_grids_das_veryfast(permuted_indexes_i, grid_i, group_maps)
                # print(assert_frame_equal(self.map_index_shuffles_to_grids_das_veryfast(permuted_indexes_i, grid_i, group_maps)[0],
                #                          permuted_files[grid_i_name][0], check_dtype=False))

                # y_row_ind = 1 # Index of row to check
                # permut_i = 0 # Permutation number
                # self.test_correspondance(y_row_ind, permut_i, permuted_indexes_i, dialogue_i,
                #                          group_maps, grid_i, grid_i_name, permuted_files)

            else:

                # print('Orig grid: ', grid_i.columns)
                # print('Orig grid: ', grid_i.iloc[0])
                permuted_files[grid_i_name] = self.map_index_shuffles_to_grids_veryfast(permuted_indexes_i, grid_i)
                # print(assert_frame_equal(self.map_index_shuffles_to_grids_veryfast(permuted_indexes_i, grid_i)[0], permuted_files[grid_i_name][0], check_dtype=False))

                # print('First permuted grid type: ', type(permuted_files[grid_i_name][0]))
                # print('First permut, first row index: ', permuted_indexes_i[0][0])
                # print('First permuted grid first list: ', permuted_files[grid_i_name][0][0])
                # print('First permut, first row index: ', permuted_indexes_i[0][1])
                # print('First permuted grid first list: ', permuted_files[grid_i_name][0][1])
                # print('Original grid row ', permuted_indexes_i[0][1], ' :', grid_i.iloc[permuted_indexes_i[0][1]])

            # break

        # print('Permutation 0 indexes: ', permuted_indexes_i[0])

        if return_originals:
            return permuted_files, original_files
        else:
            return permuted_files

    def generate_grids_for_insertion(self, folder_name='shuffled', corpus_name ='Switchboard',
                                folder_path='data/', corpus_dct=None,
                                only_grids = None, df=False, saliency=1, return_originals=False):

        shuff_path = folder_path + corpus_name + '/' + folder_name + '/'

        permuted_files = {}
        grid_names = [x for x in self.grids if not re.match(r'.+\_s[0-9][0-9]*', x) and x!='Params']
        # print('Grid names: ', grid_names)

        if return_originals:
            original_files = {}


        if only_grids is not None:
            grid_names = [x for x in grid_names if x in only_grids and x!='.DS_Store']

        print('Len grids to permute: ', len(grid_names))
        self.update_grids_dct(grid_names)

        # Permute already generated grids according to shuffled indexes order
        for grid_i_name in tqdm.tqdm(grid_names):
            # print('Grid id: ', grid_i_name)
            grid_i = self.grids.get(grid_i_name)

            # Check saliency
            if saliency>1:
                grid_i.drop([col for col in grid_i if len([i for i in grid_i[col] if i != '_']) < saliency], axis=1)

            if return_originals:
                original_files[grid_i_name] = grid_i

            shuffled_indexes_i = pd.read_csv(shuff_path+grid_i_name+'.csv', header=None, engine="c").T

            # Get permutations
            permuted_indexes_i = [[ind for ind in shuffled_indexes_i[col]] for col in shuffled_indexes_i.columns]

            # Read Params of source grids: if 'group_by' : 'DAspan' (one more layer of mapping) or 'turns' (you can get it directly)
            if self.grids_params.group_by[1] != 'turns':
                dialogue_i = corpus_dct.get(grid_i_name)
                group_maps = self.get_turns_to_da_map(dialogue_i, grid_i_name)  # list of turns span
                # print('Group maps: ', group_maps)

                # turns_number, grid, group_maps
                permuted_files[grid_i_name] = self.map_index_reinsertion_to_grids_das_veryfast(permuted_indexes_i, grid_i, group_maps)

                # y_row_ind = 1 # Index of row to check
                # permut_i = 0 # Permutation number
                # self.test_correspondance(y_row_ind, permut_i, permuted_indexes_i, dialogue_i,
                #                          group_maps, grid_i, grid_i_name, permuted_files)

            else:

                permuted_files[grid_i_name] = self.map_index_reinsertion_to_grids_veryfast(grid_i)


        # print('Permutation 0 indexes: ', permuted_indexes_i[0])

        if return_originals:
            return permuted_files, original_files
        else:
            return permuted_files


    def write_shuffled_grids(self, permuted_files, grids_path):

        print('Writing shuffled grid to: ', grids_path)
        for grid_i, permuted_files_i in tqdm.tqdm(iteritems(permuted_files)):
            # print('Grid: ', grid_i)
            # print('Perm files: ', len(permuted_files_i))
            for perm_i, perm_file in enumerate(permuted_files_i):
                # print("Writing ", grids_path+grid_i+"_s"+str(perm_i))
                # print('Test: ')
                # for i in range(1, len(perm_file)):
                #     print('N:', perm_file[i].name, [y for y in perm_file[i]])
                dump_csv(grids_path+grid_i+"_s"+str(perm_i), perm_file)
                # break

        return


def parse():
    parser = argparse.ArgumentParser(description='Shuffle generator')
    parser.add_argument('-gs', '--generate_shuffle', default='Oasis', help='Generate shuffle')
    parser.add_argument('-m', '--grid_mode', default='egrid_-coref', help='Grid mode')
    parser.add_argument('-sn', '--shuffles_number', default=20, help='Number of shuffles')
    parser.add_argument('-rr', '--rewrite_new', default=True, help='Overwrite shuffle indexes')
    args = parser.parse_args()
    return args


def run(args):
    corpus_name, grid_mode, shuffles_number, rewrite_new = args.generate_shuffle, args.grid_mode, \
                                                           args.shuffles_number, args.rewrite_new

    if args.generate_shuffle:
        grids_path = 'data/' + corpus_name + '/' + grid_mode + '/'
        corpus_dct, _ = get_corpus(corpus_name)
        grid_loader = GridLoader(grids_path)
        grid_generator = GridGenerator(coref='no_coref',
                                       nlp='no_nlp')
        grid_shuffler = GridShuffler(grids_path,
                                     grid_loader=grid_loader,
                                     grid_generator=grid_generator)
        grid_shuffler.create_shuffle_index_files(corpus_dct,
                                                 corpus_name=corpus_name,
                                                 shuffles_number=shuffles_number,
                                                 grids_path=grids_path,
                                                 rewrite_new=rewrite_new)


# def main():
#     corpus_name = 'Oasis'
#
#     # swda = Switchboard('../../Datasets/Switchboard/data/switchboard1-release2/')
#     corpus_dct, corpus_loader = get_corpus(corpus_name)
#     # print('Corpus files number: ', len(corpus_dct))
#
#     # grids_path = 'data/Switchboard/egrid_-coref/'
#     grids_path = 'data/'+corpus_name+'/egrid_-coref/'
#
#     grid_loader = GridLoader(grids_path)
#     grid_generator = GridGenerator(coref='no_coref', nlp='no_nlp')
#     grid_shuffler = GridShuffler(grids_path, grid_loader=grid_loader, grid_generator=grid_generator)
#
#
#     # Testing mode
#     # corpus_dct = {k:v for k,v in corpus_dct.iteritems() if k in ['sw_0915_3624.utt']}
#
#     grid_shuffler.create_shuffle_index_files(corpus_dct, corpus_name =corpus_name, shuffles_number=20, grids_path=grids_path)
#
#     # permuted_files = grid_shuffler.generate_shuffled_grids(corpus_dct=corpus_dct)
#     # grid_shuffler.write_shuffled_grids(permuted_files, shuff_path)


if __name__ == '__main__':
    args = parse()
    run(args)



