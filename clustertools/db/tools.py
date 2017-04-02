"""Database tools."""
import os.path as path
import StringIO as _strio
from natsort import natsorted
from glob import glob
import logging
import PIL as _PIL
import PIL.Image as _Image
import numpy as _np
import h5py as _h5py
from enum import Enum as _Enum
from multiprocessing import Pool
import tqdm
import tensorflow as tf


LOGGER = logging.getLogger(__name__)

SPECTYPES = _Enum('SPECTYPES', 'manual text implain imlossless imlossy')
VARSTRDT = _h5py.special_dtype(vlen=str)
VARBINDT = _h5py.special_dtype(vlen=_np.uint8)


def tf_decode_webp(inp_tensor):
    def _tf_decode_webp(inp):
        tmpin = _strio.StringIO(bytearray(inp))
        image = _np.array(_Image.open(tmpin))
        tmpin.close()
        return image, image.shape[0], image.shape[1]
    im, height, width = tf.py_func(_tf_decode_webp,
                                   [inp_tensor],
                                   [tf.uint8, tf.int64, tf.int64],
                                   stateful=False)
    return tf.reshape(im, tf.pack([height, width, 3]))


def create_h5_dtype(tablespec):
    """Create the appropriate HDF5 dtype for storing images and metadata.

    :param tablespec: list(tuple).
      Every tuple provides the specification on how to encode a column. The
      first tuple entry is column name, the second tells, what kind of
      specification follows (see SPECTYPES). The rest is specific: for
      `manual`, a dtype to set; `text` variable length text, nothing required;
      `implain`, direct encoding for an array, the array shape; `imlossless`,
      lossless image encoding; `imlossy`, lossy image encoding.

    :returns: The table dtype.
    """
    assert len(tablespec) > 0
    dtype_list = []
    for spec in tablespec:
        if spec[1] == SPECTYPES.manual:
            assert len(spec) == 3
            dtype_list.append((spec[0], spec[2]))
        elif spec[1] == SPECTYPES.text:
            assert len(spec) == 2
            dtype_list.append((spec[0], VARSTRDT))
        elif spec[1] == SPECTYPES.implain:
            assert len(spec) == 3
            dtype_list.append((spec[0], _np.uint8, spec[2]))
        elif spec[1] in [SPECTYPES.imlossless, SPECTYPES.imlossy]:
            assert len(spec) == 2
            dtype_list.append((spec[0] + ':webp', VARBINDT))
        else:
            raise Exception("Unknown spectype!")
    return _np.dtype(dtype_list)


def encode(tspec_entry):
    result = []
    for spec, colval in tspec_entry:
        if spec[1] == SPECTYPES.manual:
            result.append(colval)
        elif spec[1] == SPECTYPES.text:
            result.append(str(colval))
        elif spec[1] == SPECTYPES.implain:
            assert colval.dtype == _np.uint8
            result.append(colval)
        elif spec[1] in [SPECTYPES.imlossless, SPECTYPES.imlossy]:
            assert colval.ndim == 3 and colval.shape[2] == 3, (
                "Only 3-channel images are supported.")
            output = _strio.StringIO()
            pilim = _Image.fromarray(colval)
            pilim.save(output, format='WEBP',
                       lossless=(spec[1]==SPECTYPES.imlossless))
            stringrep = output.getvalue()
            output.close()
            imdata = _np.array(bytearray(stringrep)).astype('uint8')
            result.append(imdata)
        else:
            raise Exception("Unknown spectype!")
    return tuple(result)



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def encode_tf(tspec_entry):
    result = {}
    for spec, colval in tspec_entry:
        assert ':' not in spec[0] and '.' not in spec[0]
        if spec[1] == SPECTYPES.manual or spec[1] == SPECTYPES.text:
            result[spec[0]] = _bytes_feature(colval)
        elif spec[1] == SPECTYPES.implain:
            assert colval.dtype == _np.uint8
            result[spec[0] + ':plain'] = _bytes_feature(colval.tostring())
            result[spec[0] + '.height'] = _int64_feature(colval.shape[0])
            result[spec[0] + '.width'] = _int64_feature(colval.shape[1])
        elif spec[1] in [SPECTYPES.imlossless, SPECTYPES.imlossy]:
            assert colval.ndim == 3 and colval.shape[2] == 3, (
                "Only 3-channel images are supported.")
            output = _strio.StringIO()
            pilim = _PIL.Image.fromarray(colval)
            pilim.save(output, format='WEBP',
                       lossless=(spec[1]==SPECTYPES.imlossless))
            stringrep = output.getvalue()
            output.close()
            result[spec[0] + ':webp'] = _bytes_feature(stringrep)
        else:
            raise Exception("Unknown spectype!")
    return result

class TFRecordCreator(object):

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
                tff = tf.python_io.TFRecordWriter(self.dbfp + '_p%d.tfrecords' % (p_idx))
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


class H5Creator(object):

    """Helper class to parallelize HDF5 database creation."""

    def __init__(self, tablespec):
        self.h5f = None
        #self.dt = create_h5_dtype(tablespec)
        self.tablespec = tablespec

    def open(self, fp):
        self.h5f = _h5py.File(fp, mode='w')

    def close(self):
        self.h5f.flush()
        self.h5f.close()
        self.h5f = None

    def add_to_dset(self,
                    dset_name,
                    dset_func,
                    arg_list,
                    num_threads=1,
                    progress=False):
        assert num_threads > 0
        assert self.h5f is not None
        if dset_name not in self.h5f:
            dset = self.h5f.create_group(dset_name)
            for tbldesc in self.tablespec:
                dset.create_dataset(tbldesc[0],
                                    shape=(0,),
                                    maxshape=(None,),
                                    dtype=create_h5_dtype([tbldesc]),
                                    shuffle=True,
                                    compression='gzip',
                                    compression_opts=3)
        else:
            dset = self.h5f[dset_name]
        pool = Pool(processes=num_threads)
        if progress:
            pbar = tqdm.tqdm(total=len(arg_list))
        pos = 0
        while pos < len(arg_list):
            results = pool.map(dset_func, arg_list[pos:pos+num_threads])
            filtered_results = [zip(self.tablespec, result)
                                for res_list in results
                                for result in res_list]
            results = pool.map(encode, filtered_results)
            for result in results:
                for res_idx, tbldesc in enumerate(self.tablespec):
                    dset[tbldesc[0]].resize((len(dset[tbldesc[0]]) + 1,))
                    dset[tbldesc[0]][-1] = (result[res_idx],)
            self.h5f.flush()
            pos += num_threads
            if progress:
                pbar.update(num_threads)
        if progress:
            pbar.close()
            
    def __del__(self):
        if self.h5f is not None:
            self.h5f.close()

    #5, gzip, 3, webp-lossless, 498.525887966, 347.273881912, 15.3179931641
    #5, gzip, 3, webp-lossy, 57.905230999, 56.3329782486, 12.044752121
    #5, gzip, 3, internal, 63.4203698635, 678.15749073, 7.26927685738


def parse_dset(dataset):
    column_names = dataset.keys()
    nsamples = len(dataset[column_names[0]])
    for column_name in column_names:
        assert len(dataset[column_name]) == nsamples
        assert len(dataset[column_name].shape) == 1, (
            "Only 1D tables are supported! DB shape: %s." % dataset.shape)
    LOGGER.info("Found %d rows, %d columns.", nsamples, len(column_names))
    LOGGER.info("Columns:")
    coltypes = []
    for col_name in column_names:
        col_dtype = dataset[col_name].dtype  # tbl_dt[col_idx]
        if ":" in col_dtype.names[0]:  # col_name:
            if len(col_dtype.shape) == 0 and col_dtype.type == _np.void:
                # Assume this is an encoded image.
                enc_type = col_dtype.names[0][col_dtype.names[0].find(":")+1:]
                coltypes.append(enc_type)
            else:
                coltypes.append('text')
        elif col_dtype.type == _np.void and len(col_dtype.shape) in [2, 3]:
            if len(col_dtype.shape) == 2 or (
                    len(col_dtype.shape) == 3 and col_dtype.shape[2] == 3):
                coltypes.append('plain')
            else:
                coltypes.append('text')
        else:
            coltypes.append('text')
        LOGGER.info("  %s: %s", col_name, coltypes[-1])
    return nsamples, column_names, coltypes


def parse_tfdset(entry):
    column_names = entry.features.feature.keys()
    coltypes = []
    colnames = []
    for col_name in column_names:
        if '.' in col_name:
            continue
        if ":" in col_name:  # col_name:
            enc_type = col_name[col_name.find(":")+1:]
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
    result = []
    for col_idx in range(len(column_names)):
        if coltypes[col_idx] == 'text':
            result.append(str(entry.features.feature[column_names[col_idx]].bytes_list.value[0]))
        elif coltypes[col_idx] == 'plain':
            height = int(entry.features.feature[column_names[col_idx] + '.height']
                         .int64_list
                         .value[0])
            width = int(entry.features.feature[column_names[col_idx] + '.width']
                        .int64_list
                        .value[0])
            img_string = (entry.features.feature['image_raw']
                          .bytes_list
                          .value[0])
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((height, width, -1))
            result.append(entry)
        elif coltypes[col_idx] in ['png', 'jpg', 'jpeg', 'webp']:
            tmpin = _strio.StringIO(bytearray(
                entry.features.feature[column_names[col_idx]].bytes_list.value[0]))
            image = _np.array(_Image.open(tmpin))
            tmpin.close()
            result.append(image)
    if as_list:
        return result
    else:
        resdict = dict()
        for cn, rs in zip(column_names, result):
            resdict[cn] = rs
        return resdict


def decode_tf_tensors(entry, column_names, coltypes, as_list=False):
    result = []
    feat_dict = dict()
    for coln, colt in zip(column_names, coltypes):
        if '.' not in coln:
            feat_dict[coln] = tf.FixedLenFeature([], tf.string)
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
            continue
            result.append(features[coln])
        elif coltypes[col_idx] == 'plain':
            height = tf.cast(features[coln[:coln.find(":")] + ".height"],
                             tf.int64)
            width = tf.cast(features[coln[:coln.find(":")] + ".width"],
                            tf.int64)
            image = tf.decode_raw(features[coln], tf.uint8)
            image_shape = tf.pack([height, width, 3])
            result.append(tf.reshape(image, image_shape))
        elif coltypes[col_idx] == 'png':
            result.append(tf.decode_png(features[coln]))
        elif coltypes[col_idx] in ['jpg', 'jpeg']:
            result.append(tf.decode_jpeg(features[coln]))
        elif coltypes[col_idx] == 'webp':
            result.append(tf_decode_webp(features[coln]))
        else:
            raise Exception("Unsupported column type: %s!" % (coltypes[col_idx]))
        proc_names.append(coln)
    if as_list:
        return result
    else:
        resdict = dict()
        for cn, rs in zip(proc_names, result):
            resdict[cn] = rs
        return resdict


def decode(dataset, row_idx, column_names, coltypes, as_list=False):
    result = []
    for col_idx in range(len(column_names)):
        try:
            entry = dataset[column_names[col_idx]][row_idx][0]
        except Exception as ex:
            LOGGER.critical("Can't read entry with idx %d!", row_idx)
            raise ex
        if coltypes[col_idx] == 'text':
            result.append(str(entry))
        elif coltypes[col_idx] == 'plain':
            result.append(entry)
        elif coltypes[col_idx] in ['png', 'jpg', 'jpeg', 'webp']:
            tmpin = _strio.StringIO(bytearray(entry))
            image = _np.array(_Image.open(tmpin))
            tmpin.close()
            result.append(image)
    if as_list:
        return result
    else:
        return {(cn, rs) for cn, rs in zip(column_names, result)}


class H5Converter(object):

    """Helper class to parallelize HDF5 database conversion."""

    def __init__(self, tablespec):
        self.h5f_in = None
        self.h5f_out = None
        self.tablespec = tablespec

    def open(self, in_fp, out_fp):
        self.h5f_in = _h5py.File(in_fp, mode='r')
        self.h5f_out = _h5py.File(out_fp, mode='w')

    def close(self):
        self.h5f_in.close()
        self.h5f_in = None
        self.h5f_out.flush()
        self.h5f_out.close()
        self.h5f_out = None

    def convert_dset(self,
                     dset_in_name,
                     dset_name,
                     dset_func,
                     num_threads=1,
                     progress=False):
        assert num_threads > 0
        assert self.h5f_in is not None and self.h5f_out is not None
        if dset_name not in self.h5f_out:
            dset = self.h5f_out.create_group(dset_name)
            for tbldesc in self.tablespec:
                dset.create_dataset(tbldesc[0],
                                    shape=(0,),
                                    maxshape=(None,),
                                    dtype=create_h5_dtype([tbldesc]),
                                    shuffle=True,
                                    compression='gzip',
                                    compression_opts=3)
        else:
            dset = self.h5f_out[dset_name]
        dset_in = self.h5f_in[dset_in_name]
        nsamples, colinnames, colintypes = parse_dset(dset_in)
        pool = Pool(processes=num_threads)
        if progress:
            pbar = tqdm.tqdm(total=nsamples)
        pos = 0
        while pos < nsamples:
            inputs = [decode(dset_in, idx, colinnames, colintypes)
                      for idx in range(pos, min(pos+num_threads, nsamples))]
            results = pool.map(dset_func, inputs)
            filtered_results = [zip(self.tablespec, result)
                                for res_list in results
                                for result in res_list]
            results = pool.map(encode, filtered_results)
            for result in results:
                for res_idx, tbldesc in enumerate(self.tablespec):
                    dset[tbldesc[0]].resize((len(dset[tbldesc[0]]) + 1,))
                    dset[tbldesc[0]][-1] = (result[res_idx],)
            self.h5f_out.flush()
            pos += num_threads
            if progress:
                pbar.update(num_threads)
        if progress:
            pbar.close()
            
    def __del__(self):
        if self.h5f_in is not None:
            self.h5f_in.close()
        if self.h5f_out is not None:
            self.h5f_out.close()


class TFConverter(object):

    """Helper class to parallelize tfrecord database conversion."""

    def __init__(self, tablespec):
        self.tff_in = None
        self.tff_out = None
        self.tablespec = tablespec

    def open(self, in_fp, out_fp):
        assert not in_fp.endswith(".tfrecords"), ("Please specify the db name "
                                                  "without the file ending.")
        assert not out_fp.endswith(".tfrecords"), ("Please specify the db name "
                                                   "without the file ending.")
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
        LOGGER.info("Converting the following records: %s.", str(records))
        pool = Pool(processes=num_threads)
        initialized = False
        p_idx = 0
        for recfile in records:
            LOGGER.info("Processing `%s`...", recfile)
            record_iterator = tf.python_io.tf_record_iterator(path=recfile)
            tff = tf.python_io.TFRecordWriter(self.tff_out + '_p%d.tfrecords' % (p_idx))
            p_idx += 1
            file_complete = False
            if progress:
                pbar = tqdm.tqdm()
            while not file_complete:
                inputs = []
                if not initialized:
                    string_record = record_iterator.next()
                    example = tf.train.Example()
                    example.ParseFromString(string_record)
                    # Parse fields.
                    colnames, coltypes = parse_tfdset(example)
                    initialized = True
                    inputs.append(example)
                while len(inputs) < num_threads:
                    try:
                        string_record = record_iterator.next()
                    except:
                        file_complete = True
                        break
                    example = tf.train.Example()
                    example.ParseFromString(string_record)
                    inputs.append(example)
                inputs = [decode_tf(example, colnames, coltypes, as_list=False)
                          for example in inputs]
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
