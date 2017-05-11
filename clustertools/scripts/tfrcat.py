#!/usr/bin/env python2
import os
import os.path as path
from natsort import natsorted
from glob import glob
import numpy as np
import logging
import scipy.misc as sm
import click
import tqdm
import tensorflow as tf
import clustertools.db.tools as cdbt
from clustertools.log import LOGFORMAT


LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("db_fp", type=click.Path())
@click.option("--out_fp", type=click.Path(writable=True, exists=False),
              default=None,
              help="Output directory. If unspecified, use `{db_fp}.content`.")
def cli(db_fp, out_fp=None):
    """Output a database (without .tfrecord and _p ending) to a directory."""
    if out_fp is None:
        out_fp = db_fp + '.content'
    LOGGER.info("Writing from DB `%s` to `%s`.", db_fp, out_fp)
    records = natsorted(glob(db_fp + '_p*.tfrecords'))
    if not path.exists(out_fp):
        os.mkdir(out_fp)
    nsamples, colnames, coltypes = cdbt.scan_tfdb(records)
    fillwidth = int(np.ceil(np.log10(nsamples)))
    run_idx = 0
    pbar = tqdm.tqdm(total=nsamples)
    for rec_fp in records:
        record_iterator = tf.python_io.tf_record_iterator(path=rec_fp)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            values = cdbt.decode_tf(example, colnames, coltypes, as_list=True)
            for colt, coln, val in zip(coltypes, colnames, values):
                if colt == 'text':
                    with open(path.join(out_fp,
                                        '{0:0{width}}_{1}.txt'.format(
                                            run_idx, coln, width=fillwidth)
                                        .encode("ascii")),
                              'w') as outf:
                        outf.write(val)
                elif colt == 'plain':
                    np.save(
                        path.join(out_fp, '{0:0{width}}_{1}.npy'.format(
                            run_idx, coln, width=fillwidth)),
                        val)
                else:
                    sm.imsave(
                        path.join(out_fp, '{0:0{width}}_{1}.png'.format(
                            run_idx, coln, width=fillwidth)),
                        val)
            run_idx += 1
            pbar.update(1)
    pbar.close()
    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    cli()
