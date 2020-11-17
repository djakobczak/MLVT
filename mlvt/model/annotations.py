import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


LOG = logging.getLogger('MLVT')


def label_samples(src_csv_file, dst_csv_file, paths, label):
    src_csv = pd.read_csv(src_csv_file)
    imgs_df = src_csv.loc[src_csv.paths.isin(paths)]
    idxs = imgs_df.index.tolist()
    src_csv.drop(idxs, inplace=True)
    src_csv.to_csv(src_csv_file, index=False)
    # imgs = imgs_df.values.astype(str)

    with open(dst_csv_file, 'a') as f:
        for path in paths:
            f.write(f'{path},{label}\n')


def create_csv_file(target_file, data_dir, header='paths,labels',
                    force=True, label=None, mode="w+"):
    if force:
        target_dir, _ = os.path.split(target_file)
        Path(target_dir).mkdir(parents=True, exist_ok=True)
    try:
        with open(target_file, mode) as tf:
            if header:
                tf.write(f'{header}\n')

            if data_dir is None:
                return

            if os.path.isdir(data_dir):
                LOG.info(f'Start loading from {data_dir}...')
                for f in tqdm(os.listdir(data_dir)):
                    feature_rpath = os.path.join(data_dir, f)
                    line = f'{feature_rpath}\n' if label is None \
                        else f'{feature_rpath},{label}\n'
                    tf.write(line)
        LOG.info(f'Created file {target_file}')
    except OSError as e:
        return e
