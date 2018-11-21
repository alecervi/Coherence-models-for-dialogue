from __future__ import print_function
from load_grids import GridLoader
import pandas as pd
import os
import tqdm


def main():
    corpus = 'AMI'
    grids_path = 'data/'+corpus+'/egrid_-coref_DAspan_da_noentcol/'
    no_ents_baseline_path = 'data/'+corpus+'/noents_baseline/'
    grid_loader = GridLoader(grids_path)
    if not os.path.exists(grids_path):
        raise TypeError("The following folder does not exist " + grids_path)

    grids, _ = grid_loader.load_data()
    print('Number of grids: ', len(grids))
    grid_names = [x for x in grids if x != 'Params']

    for grid_i_name in tqdm.tqdm(grid_names):
        grid_i = grids.get(grid_i_name)
        grid_i_da_seq = [list(set([da for da in row if da != '_'])) for index, row in grid_i.iterrows()]
        if all(len(das) == 1 for das in grid_i_da_seq):
            grid_i_da_seq = [da[0] for da in grid_i_da_seq]
            df = pd.DataFrame.from_items([('all_das', grid_i_da_seq)])
            df.to_csv(path_or_buf=no_ents_baseline_path + grid_i_name + '.csv', index=False)
        else:
            raise TypeError("Not only one Dialogue Act per row in grid " + grid_i_name)



if __name__ == '__main__':
    main()