"""Database tools."""
import os.path as path
import StringIO as _strio
from natsort import natsorted
from glob import glob
import logging
import PIL as _PIL
import PIL.Image as _Image
import numpy as _np
from aenum import Enum as _Enum
from multiprocessing import Pool
import tqdm
import tensorflow as tf


LOGGER = logging.getLogger(__name__)

SPECTYPES = _Enum('SPECTYPES', 'manual text implain imlossless imlossy')


def tf_decode_webp(inp_tensor):
    """Decode a webp encoded image."""
    def _tf_decode_webp(inp):
        tmpin = _strio.StringIO(bytearray(inp))
        image = _np.array(_Image.open(tmpin))
        tmpin.close()
        return image, image.shape[0], image.shape[1]
    im, height, width = tf.py_func(_tf_decode_webp,
                                   [inp_tensor],
                                   [tf.uint8, tf.int64, tf.int64],
                                   stateful=False)
    return tf.reshape(im, tf.cast(tf.pack([height, width, 3]), tf.int32))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def encode_tf(tspec_entry):
    """Encode an entry to tensorflow format."""
    result = {}
    for spec, colval in tspec_entry:
        assert ':' not in spec[0] and '.' not in spec[0]
        if spec[1] == SPECTYPES.manual or spec[1] == SPECTYPES.text:
            result[spec[0]] = _bytes_feature(colval)
        elif spec[1] == SPECTYPES.implain:
            #assert colval.dtype == _np.uint8
            result[spec[0] + ':plain:' + colval.dtype.name] = \
                _bytes_feature(colval.tostring())
            if colval.ndim < 1:
                result[spec[0] + '.height'] = _int64_feature(1)
            else:
                result[spec[0] + '.height'] = _int64_feature(colval.shape[0])
            if colval.ndim < 2:
                result[spec[0] + '.width'] = _int64_feature(1)
            else:
                result[spec[0] + '.width'] = _int64_feature(colval.shape[1])
        elif spec[1] == SPECTYPES.imlossless:
            assert colval.ndim == 3 and colval.shape[2] == 3, (
                "Only 3-channel images are supported (%s)." % (spec[0]))
            output = _strio.StringIO()
            pilim = _PIL.Image.fromarray(colval)
            pilim.save(output, format='PNG')
            stringrep = output.getvalue()
            output.close()
            result[spec[0] + ':png'] = _bytes_feature(stringrep)
        elif spec[1] == SPECTYPES.imlossy:
            assert colval.ndim == 3 and colval.shape[2] == 3, (
                "Only 3-channel images are supported.")
            output = _strio.StringIO()
            pilim = _PIL.Image.fromarray(colval)
            pilim.save(output, format='JPEG')
            stringrep = output.getvalue()
            output.close()
            result[spec[0] + ':jpg'] = _bytes_feature(stringrep)
        elif spec[1] in [SPECTYPES.imlossless, SPECTYPES.imlossy] and False:
            assert colval.ndim == 3 and colval.shape[2] == 3, (
                "Only 3-channel images are supported.")
            output = _strio.StringIO()
            pilim = _PIL.Image.fromarray(colval)
            pilim.save(output, format='WEBP',
                       lossless=(spec[1] == SPECTYPES.imlossless))
            stringrep = output.getvalue()
            output.close()
            result[spec[0] + ':webp'] = _bytes_feature(stringrep)
        else:
            raise Exception("Unknown spectype!")
    return result


class TFRecordCreator(object):

    """Helper class for parallel construction of .tfrecord databases."""

    def __init__(self, tablespec, examples_per_file=1000):
        self.tablespec = tablespec
        self.examples_per_file = examples_per_file
        self.dbfp = None

    def open(self, fp):
        assert not fp.endswith(".tfrecords"), ("Please specify the filename wo"
                                               "file ending (.tfrecords).")
        self.dbfp = fp
        assert not path.exists(fp + '_p0.tfrecords'), ("Dataset exists!")

    def close(self):
        self.dbfp = None

    def add_to_dset(self,
                    dset_func,
                    arg_list,
                    num_threads=1,
                    progress=False):
        assert num_threads > 0
        assert self.dbfp is not None
        pool = Pool(processes=num_threads)
        if progress:
            pbar = tqdm.tqdm(total=len(arg_list))
        tff = None
        p_idx = -1
        p_pos = 0
        pos = 0
        while pos < len(arg_list):
            if p_pos > self.examples_per_file or p_idx == -1:
                if tff is not None:
                    tff.close()
                p_idx += 1
                tff = tf.python_io.TFRecordWriter(self.dbfp +
                                                  '_p%d.tfrecords' % (p_idx))
                p_pos = 0
            if num_threads == 1:
                results = map(dset_func, arg_list[pos:pos+num_threads])
            else:
                results = pool.map(dset_func, arg_list[pos:pos+num_threads])
            filtered_results = [zip(self.tablespec, result)
                                for res_list in results
                                for result in res_list]
            if num_threads == 1:
                results = map(encode_tf, filtered_results)
            else:
                results = pool.map(encode_tf, filtered_results)
            for result in results:
                example = tf.train.Example(
                    features=tf.train.Features(feature=result))
                tff.write(example.SerializeToString())
                p_pos += 1
            pos += num_threads
            if progress:
                pbar.update(num_threads)
        if progress:
            pbar.close()
        tff.close()

    def __del__(self):
        self.dbfp = None


def parse_tfdset(entry):
    """Parse a dataset entry for columns and types."""
    column_names = entry.features.feature.keys()
    coltypes = []
    colnames = []
    for col_name in column_names:
        if '.' in col_name:
            continue
        if ":" in col_name:  # col_name:
            enc_type = col_name.split(":")[1]
            coltypes.append(enc_type)
        else:
            coltypes.append('text')
        colnames.append(col_name)
    LOGGER.info("Found %d columns.", len(column_names))
    LOGGER.info("Columns:")
    for col_name, col_type in zip(colnames, coltypes):
        LOGGER.info("  %s: %s", col_name, col_type)
    return colnames, coltypes


def scan_tfdb(input_paths):
    """Scan a set of tfrecord files for entry type and number of examples."""
    nsamples = 0
    LOGGER.info("Scanning input files...")
    data_def = None
    initialized = False
    for input_fp in input_paths:
        recit = tf.python_io.tf_record_iterator(input_fp)
        for string_record in recit:
            if not initialized:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                colnames, coltypes = parse_tfdset(example)
                initialized = True
                if data_def is None:
                    data_def = colnames, coltypes
                else:
                    assert colnames == data_def[0]
                    assert coltypes == data_def[1]
            nsamples += 1
    return nsamples, colnames, coltypes


def decode_tf(entry, column_names, coltypes, as_list=False):
    """Decode a tfrecord string entry."""
    result = []
    for col_idx in range(len(column_names)):
        if coltypes[col_idx] == 'text':
            result.append(str(entry.features.feature[column_names[col_idx]]
                              .bytes_list
                              .value[0]))
        elif coltypes[col_idx] == 'plain':
            coln_plain = column_names[col_idx].split(":")[0]
            dtype_string = column_names[col_idx].split(":")[2]
            height = int(entry.features.feature[coln_plain + '.height']
                         .int64_list
                         .value[0])
            width = int(entry.features.feature[coln_plain + '.width']
                        .int64_list
                        .value[0])
            img_string = (entry.features.feature[column_names[col_idx]]
                          .bytes_list
                          .value[0])
            img_1d = _np.fromstring(img_string, _np.dtype(dtype_string))
            image = img_1d.reshape((height, width, -1))
            result.append(image)
        elif coltypes[col_idx] in ['png', 'jpg', 'jpeg', 'webp']:
            tmpin = _strio.StringIO(bytearray(
                entry.features.feature[column_names[col_idx]]
                .bytes_list
                .value[0]))
            image = _np.array(_Image.open(tmpin))
            tmpin.close()
            result.append(image)
    if as_list:
        return result
    else:
        resdict = dict()
        for cn, rs in zip(column_names, result):
            if ':' in cn:
                resdict[cn[:cn.find(":")]] = rs
            else:
                resdict[cn] = rs
        return resdict


def decode_tf_tensors(entry, column_names, coltypes, as_list=False):
    """Decode a tfrecord entry to tensors."""
    result = []
    feat_dict = dict()
    for coln, colt in zip(column_names, coltypes):
        if '.' not in coln:
            feat_dict[coln] = tf.FixedLenFeature([], tf.string)
            if ':plain' in coln:
                coln_raw = coln.split(":")[0]
                feat_dict[coln_raw + '.width'] = tf.FixedLenFeature([],
                                                                    tf.int64)
                feat_dict[coln_raw + '.height'] = tf.FixedLenFeature([],
                                                                     tf.int64)
        else:
            feat_dict[coln] = tf.FixedLenFeature([], tf.int64)
    features = tf.parse_single_example(
        entry,
        features=feat_dict)
    proc_names = []
    for col_idx in range(len(column_names)):
        coln = column_names[col_idx]
        if "." in coln:
            continue
        if coltypes[col_idx] == 'text':
            result.append(features[coln])
        elif coltypes[col_idx] == 'plain':
            height = tf.cast(features[coln[:coln.find(":")] + ".height"],
                             tf.int32)
            width = tf.cast(features[coln[:coln.find(":")] + ".width"],
                            tf.int32)
            dtype_string = coln.split(":")[2]
            image = tf.decode_raw(features[coln], tf.as_dtype(dtype_string))
            image_shape = tf.stack([height, width, -1])
            result.append(tf.reshape(image, image_shape))
        elif coltypes[col_idx] == 'png':
            result.append(tf.image.decode_png(features[coln]))
        elif coltypes[col_idx] in ['jpg', 'jpeg']:
            result.append(tf.image.decode_jpeg(features[coln]))
        elif coltypes[col_idx] == 'webp':
            result.append(tf_decode_webp(features[coln]))
        else:
            raise Exception("Unsupported column type: %s!" % (
                coltypes[col_idx]))
        proc_names.append(coln)
    if as_list:
        return result
    else:
        resdict = dict()
        for cn, rs in zip(proc_names, result):
            if ':' in cn:
                resdict[cn[:cn.find(":")]] = rs
            else:
                resdict[cn] = rs
        return resdict


class TFConverter(object):

    """Helper class to parallelize tfrecord database conversion."""

    def __init__(self, tablespec):
        self.tff_in = None
        self.tff_out = None
        self.tablespec = tablespec

    def open(self, in_fp, out_fp):
        assert not in_fp.endswith(".tfrecords"), ("Please specify the db name "
                                                  "without the file ending.")
        assert not out_fp.endswith(".tfrecords"), ("Please specify the db name"
                                                   " without the file ending.")
        self.tff_in = in_fp
        self.tff_out = out_fp
        assert not path.exists(out_fp + '_p0.tfrecords')

    def close(self):
        self.tff_in = None
        self.tff_out = None

    def convert_dset(self,
                     dset_func,
                     num_threads=1,
                     progress=False):
        assert num_threads > 0
        assert self.tff_in is not None and self.tff_out is not None
        records = natsorted(glob(self.tff_in + '_p*.tfrecords'))
        nsamples, colnames, coltypes = scan_tfdb(records)
        LOGGER.info("Converting %d records from: %s.", nsamples, str(records))
        pool = Pool(processes=num_threads)
        p_idx = 0
        if progress:
            pbar = tqdm.tqdm(total=nsamples)
        for recfile in records:
            record_iterator = tf.python_io.tf_record_iterator(path=recfile)
            tff = tf.python_io.TFRecordWriter(self.tff_out +
                                              '_p%d.tfrecords' % (p_idx))
            p_idx += 1
            file_complete = False
            while not file_complete:
                inputs = []
                while len(inputs) < num_threads:
                    try:
                        string_record = record_iterator.next()
                    except:
                        file_complete = True
                        break
                    example = tf.train.Example()
                    example.ParseFromString(string_record)
                    inputs.append(example)
                inputs = [decode_tf(exin, colnames, coltypes, as_list=False)
                          for exin in inputs]
                if num_threads == 1:
                    results = map(dset_func, inputs)
                else:
                    results = pool.map(dset_func, inputs)
                filtered_results = [zip(self.tablespec, result)
                                    for res_list in results
                                    for result in res_list]
                if num_threads == 1:
                    results = map(encode_tf, filtered_results)
                else:
                    results = pool.map(encode_tf, filtered_results)
                for result in results:
                    example = tf.train.Example(
                        features=tf.train.Features(feature=result))
                    tff.write(example.SerializeToString())
                if progress:
                    pbar.update(num_threads)
        if progress:
            pbar.close()

    def __del__(self):
        self.tff_in = None
        self.tff_out = None
