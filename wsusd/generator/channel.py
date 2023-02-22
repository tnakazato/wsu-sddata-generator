import functools
import numpy as np
import os
import scipy
import shutil

from casatasks.private import sdutil
from casatools import msmetadata

from _logging import get_logger

logger = get_logger(__name__)


@functools.lru_cache()
def gauss_normalized(n, sigma):
    gauss = scipy.signal.windows.gaussian(n, sigma)
    gauss /= gauss.sum()
    return gauss


def robust_stddev(data, clipthresh=3, clipniter=3):
    # compute robust stddev using n-sigma clipping
    mask = np.zeros(len(data), dtype=bool)
    stddev = np.nan
    for _ in range(clipniter + 1):
        marr = np.ma.masked_array(data, mask)
        stddev = np.ma.std(marr)
        thresh = stddev * clipthresh
        _m = np.abs(data) > thresh
        mask = np.logical_or(mask, _m)

    return stddev


def add_noise(data, noise_stddev):
    generator = np.random.default_rng()
    noise = generator.normal(0, noise_stddev, len(data))
    return data + noise


def interpolate_data_single(data, cf_in, cf_out, sigma=2):
    # data should be one dimensional
    assert len(data.shape) == 1

    # nchan should be equal to len(cf_in)
    assert len(data) == len(cf_in)

    nchan = len(data)
    n = min(nchan, sigma * 10)
    gauss = gauss_normalized(n, sigma)

    smoothed = np.convolve(data, gauss, mode='same')

    _interpolator = scipy.interpolate.interp1d(
        cf_in, smoothed, bounds_error=False, fill_value=(smoothed[0], smoothed[-1]))
    interpolated = _interpolator(cf_out)

    diff = data - smoothed
    noise_std = robust_stddev(diff, clipthresh=3, clipniter=3)
    logger.debug(f'native std {diff.std()} robust std {noise_std}')
    corrupted = add_noise(interpolated, noise_std)

    noise_in = data - smoothed
    logger.debug(f'noise(in) {noise_in[:5]} mean {noise_in.mean()} std {noise_in.std()} med {np.median(noise_in)}')
    noise = corrupted - interpolated
    logger.debug(f'noise(out) {noise[:5]} mean {noise.mean()} std {noise.std()} med {np.median(noise)}')

    return corrupted


def interpolate_data(data_in, cf_in, cf_out, sigma=2):
    # data.shape should be (npol, nchan, nrow)
    assert len(data_in.shape) == 3

    # channel frequencies should be one-dimensional array
    assert len(cf_in.shape) == 1
    assert len(cf_out.shape) == 1

    npol, _, nrow = data_in.shape
    nchan = len(cf_out)

    logger.debug(f'cf_in.shape = {cf_in.shape}')
    logger.debug(f'cf_out.shape = {cf_out.shape}')
    logger.debug(f'data.shape = {data_in.shape}')
    data_out = np.zeros((npol, nchan, nrow), dtype=data_in.dtype)
    for ipol in range(npol):
        for irow in range(nrow):
            _data_in = data_in[ipol, :, irow]
            data_out[ipol, :, irow] = interpolate_data_single(_data_in, cf_in, cf_out, sigma)

    return data_out


def interpolate_bool(data_in, cf_in, cf_out):
    # data.shape should be (npol, nchan, nrow)
    assert len(data_in.shape) == 3

    # channel frequencies should be one-dimensional array
    assert len(cf_in.shape) == 1
    assert len(cf_out.shape) == 1

    npol, _, nrow = data_in.shape
    nchan = len(cf_out)

    data_out = np.zeros((npol, nchan, nrow), dtype=data_in.dtype)
    for ipol in range(npol):
        for irow in range(nrow):
            _data_in = data_in[ipol, :, irow]
            obj = scipy.interpolate.interp1d(cf_in, _data_in, kind='nearest', bounds_error=False, fill_value=(_data_in[0], _data_in[-1]))
            data_out[ipol, :, irow] = np.array(obj(cf_out), dtype=bool)

    return data_out


class TableUpdater:
    @property
    def columns(self):
        return []

    def taql(self, spw: int):
        return ''

    def __init__(self, vis: str, target_spws: int, table_name, **kwargs):
        self.vis = vis
        self.target_spws = [int(spw) for spw in target_spws]
        self.table_name = table_name
        for attr in ['chan_factor', 'freq_in', 'freq_out']:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

        self.data_in = None
        self.data_out = None

    def read(self):
        self.data_in = {}
        for spw in self.target_spws:
            taql = self.taql(spw)
            assert len(taql) > 0
            with sdutil.table_selector(self.table_name, taql, nomodify=True) as tb:
                if tb.nrows() == 0:
                    continue

                existing_columns = tb.colnames()
                _data_in = dict(
                    (col, tb.getcol(col)) for col in self.columns if col in existing_columns and tb.iscelldefined(col, 0)
                )

            self.data_in[spw] = _data_in

    def update(self):
        pass

    def flush(self):
        if not isinstance(self.data_out, dict):
            return

        for spw in self.target_spws:
            taql = self.taql(spw)
            assert len(taql) > 0
            with sdutil.table_selector(self.table_name, taql, nomodify=False) as tb:
                _data_out = self.data_out[spw]
                for col, val in _data_out.items():
                    tb.putcol(col, val)


class SpectralWindowUpdater(TableUpdater):
    @property
    def columns(self):
        return ['CHAN_FREQ', 'CHAN_WIDTH', 'NUM_CHAN', 'EFFECTIVE_BW', 'RESOLUTION']

    def taql(self, spw):
        # inside TaQL, ROWNUMBER returns 1-based row number while spw is 0-based
        return f'ROWNUMBER() == {spw + 1}'

    def __init__(self, vis: str, target_spws: list, chan_factor: float):
        table_name = os.path.join(vis, 'SPECTRAL_WINDOW')
        super().__init__(vis, target_spws, table_name, chan_factor=chan_factor)

    def get_chan_freq_in(self):
        if isinstance(self.data_in, dict):
            return dict((spw, data['CHAN_FREQ'][:, 0]) for spw, data in self.data_in.items())
        else:
            return None

    def get_chan_freq_out(self):
        if isinstance(self.data_out, dict):
            return dict((spw, data['CHAN_FREQ'][:, 0]) for spw, data in self.data_out.items())
        else:
            return None

    def _update_num_chan(self, spw, nchan):
        nchan_new = nchan * self.chan_factor
        logger.debug(f'spw {spw}: nchan(in) {nchan[0]}, nchan(out) {nchan_new[0]}')
        return nchan_new

    def _update_chan_width(self, spw, chan_width):
        chan_freq = self.data_in[spw]['CHAN_FREQ']
        nchan_new = self.data_out[spw]['NUM_CHAN']

        start_chan = chan_freq[0][0] - chan_width[-1][0] / 2
        end_chan = chan_freq[-1][0] + chan_width[-1][0] / 2
        bandwidth = end_chan - start_chan

        cw_new = np.zeros(nchan_new, dtype=chan_width.dtype) + (bandwidth / nchan_new)
        logger.debug(f'spw {spw}: chan_width(in) {chan_width[0][0]} chan_width(out) {cw_new[0]}')
        return cw_new[:, np.newaxis]

    def _update_chan_freq(self, spw, chan_freq):
        chan_width = self.data_in[spw]['CHAN_WIDTH']
        cw_new = self.data_out[spw]['CHAN_WIDTH']
        start_chan = chan_freq[0][0] - chan_width[-1][0] / 2
        end_chan = chan_freq[-1][0] + chan_width[-1][0] / 2
        _start = start_chan + cw_new[0][0] / 2
        _end = end_chan
        _step = cw_new[0][0]
        logger.debug(f'_start {_start} _end {_end} _step {_step}')

        cf_new = np.arange(_start, _end, _step, dtype=chan_freq.dtype)
        logger.debug(f'spw {spw}: chan_freq(in) {chan_freq[0][0]} chan_freq(out) {cf_new[0]}')
        return cf_new[:, np.newaxis]

    def _scale_by_chan_width(self, spw, data):
        cw = self.data_in[spw]['CHAN_WIDTH']
        cw_new = self.data_out[spw]['CHAN_WIDTH']
        data_new = np.abs(cw_new * data[0][0] / cw[0][0])
        return data_new

    def _no_scale(self, spw, data):
        cw_new = self.data_out[spw]['CHAN_WIDTH']
        data_new = np.zeros_like(cw_new) + abs(data[0][0])
        return data_new

    def _update_col(self, column, update_func):
        for spw in self.target_spws:
            _data_in = self.data_in[spw][column]
            _data_out = update_func(spw, _data_in)
            self.data_out[spw][column] = _data_out

    def update(self):
        self.data_out = dict((spw, {}) for spw in self.target_spws)

        # NUM_CHAN
        self._update_col('NUM_CHAN', self._update_num_chan)

        # CHAN_WIDTH
        self._update_col('CHAN_WIDTH', self._update_chan_width)

        # CHAN_FREQ
        self._update_col('CHAN_FREQ', self._update_chan_freq)

        # EFFECTIVE_BW
        # effective noise bandwidth is proportional to channel width
        self._update_col('EFFECTIVE_BW', self._scale_by_chan_width)
        # self._update_col('EFFECTIVE_BW', self._no_scale)

        # RESOLUTION
        self._update_col('RESOLUTION', self._scale_by_chan_width)


class SyscalUpdator(TableUpdater):
    @property
    def columns(self):
        return ['TCAL_SPECTRUM', 'TRX_SPECTRUM', 'TSKY_SPECTRUM', 'TSYS_SPECTRUM',
                'TANT_SPECTRUM', 'TANT_TSYS_SPECTRUM']

    def taql(self, spw):
        return f'SPECTRAL_WINDOW_ID == {spw}'

    def __init__(self, vis: str, spw: int, freq_in: dict, freq_out: dict):
        table_name = os.path.join(vis, 'SYSCAL')
        super().__init__(vis, spw, table_name, freq_in=freq_in, freq_out=freq_out)

    def update(self):
        self.data_out = dict((spw, {}) for spw in self.target_spws)

        for spw, _data_in in self.data_in.items():
            _freq_in = self.freq_in[spw]
            _freq_out = self.freq_out[spw]
            for col, arr in _data_in.items():
                logger.info(f'spw {spw}: updating column {col}')
                # scale temperature data with 1 / sqrt(chan_width)
                # _cw_in = abs(_freq_in[1] - _freq_in[0])
                # _cw_out = abs(_freq_out[1] - _freq_out[0])
                # scale_factor = np.sqrt(_cw_in / _cw_out)
                # logger.info(f'spw {spw}: temperature scaling factor is {scale_factor}')
                scale_factor = 1.0
                self.data_out[spw][col] = interpolate_data(arr, _freq_in, _freq_out) * scale_factor


def copy_table_structure(vis, outputvis):
    logger.debug(f'copying {vis} into {outputvis}')
    with sdutil.table_manager(vis) as tb:
        tout = tb.copy(outputvis, norows=True)
        tout.close()

    logger.debug('done copying')


def copy_subtable_rows(vis, outputvis):
    with sdutil.table_manager(vis) as tb:
        table_names = filter(
            lambda x: isinstance(x[1], str) and x[1].startswith('Table: '),
            tb.getkeywords().items()
        )

    for name, path in table_names:
        logger.debug(f'copying {name} rows')
        src = path[7:]
        dst = os.path.join(outputvis, name)
        with sdutil.table_manager(src) as tb:
            tb.copyrows(dst)

        logger.debug(f'done {name}')


def copy_main_columns(vis, outputvis, ignore):
    with sdutil.table_manager(outputvis, nomodify=False) as tb_out:
        nrow_out = tb_out.nrows()
        with sdutil.table_manager(vis) as tb_in:
            nrow = tb_in.nrows()
            tb_out.addrows(nrow - nrow_out)
            for col in tb_out.colnames():
                logger.debug(col)
                if col in ignore or not tb_in.iscelldefined(col, 0):
                    continue

                logger.debug(f'copy column {col}')
                data = tb_in.getcol(col)
                tb_out.putcol(col, data)
                logger.debug(f'done {col}')


def rename_table(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)

    os.rename(src, dst)


class ChunkInterpolator:
    def __init__(self, cf_in, cf_out):
        self.cf_in = cf_in
        self.cf_out = cf_out

    def __scale_data(self, data_in, factor):
        # shape of data_in should be (npol, nchan, nrow)
        assert len(data_in.shape) == 3
        assert data_in.shape[1] == len(self.freq_in)

        npol, _, nrow = data_in.shape
        nchan = len(self.freq_out)
        data_out = np.zeros((npol, nchan, nrow), dtype=data_in.dtype)
        data_out[:] = data_in[0, 0, 0] * factor
        return data_out

    def __update_weight_spectrum(self, data_in):
        # weight is proportional to channel width
        cw_in = abs(self.cf_in[1] - self.cf_in[0])
        cw_out = abs(self.cf_out[1] - self.cf_out[0])
        factor = cw_out / cw_in
        return self.__scale_data(data_in, factor)

    def __update_sigma_spectrum(self, data_in):
        # sigma is proportional to 1 / sqrt(channel width)
        cw_in = abs(self.cf_in[1] - self.cf_in[0])
        cw_out = abs(self.cf_out[1] - self.cf_out[0])
        factor = 1 / np.sqrt(cw_out / cw_in)
        return self.__scale_data(data_in, factor)

    def __call__(self, chunk):
        chunk_start, nrow_chunk, data_in = chunk

        data_out = {}

        # update data column (FLOAT_DATA/DATA)
        for column in ['FLOAT_DATA', 'DATA']:
            if column in data_in:
                data_out[column] = interpolate_data(data_in[column], self.cf_in, self.cf_out)
                break

        # update FLAG
        column = 'FLAG'
        data_out[column] = interpolate_bool(data_in[column], self.cf_in, self.cf_out)

        # update WEIGHT_SPECTRUM
        column = 'WEIGHT_SPECTRUM'
        if column in data_in:
            data_out[column] = self.__update_weight_spectrum(data_in[column])

        # update_SIGMA_SPECTRUM
        column = 'SIGMA_SPECTRUM'
        if column in data_in:
            data_out[column] = self.__update_sigma_spectrum(data_in[column])

        return chunk_start, nrow_chunk, data_out


class MainUpdater(TableUpdater):
    @functools.lru_cache(1)
    def __columns(self):
        columns = []
        with sdutil.table_manager(self.vis) as tb:
            colnames = tb.colnames()
            colname = 'FLOAT_DATA'
            if colname in colnames:
                columns.append(colname)
            else:
                columns.append('DATA')

            columns.append('FLAG')

            for colname in ['SIGMA_SPECTRUM', 'WEIGHT_SPECTRUM']:
                if colname in colnames and tb.iscelldefined(colname, 0):
                    columns.append(colname)

        return columns

    @property
    def columns(self):
        return self.__columns()

    def taql(self, spw):
        return f'DATA_DESC_ID == {self.ddid[spw]}'

    def __init__(self, vis: str, target_spws: list, freq_in: dict, freq_out: dict):
        table_name = vis
        super().__init__(vis, target_spws, table_name, freq_in=freq_in, freq_out=freq_out)

        msmd = msmetadata()
        msmd.open(self.vis)
        self.ddid = {}
        for spw in range(msmd.nspw()):
            ddids = msmd.datadescids(spw)
            if len(ddids) > 0:
                self.ddid[spw] = ddids[0]
        self.all_spws = sorted(self.ddid.keys())
        msmd.close()

        self.tmp_vis = f'genwsusd.{vis}.tmp'
        self.backup_vis = f'genwsusd.{vis}.bak'

    def _read_main(self, spw):
        nrow_chunk_default = 100

        taql = self.taql(spw)
        with sdutil.table_selector(self.vis, taql) as tb:
            nrow = tb.nrows()
            nchunk = nrow // nrow_chunk_default
            nmod = nrow % nrow_chunk_default
            chunk_list = [nrow_chunk_default] * nchunk
            if nmod > 0:
                chunk_list.append(nmod)

            chunk_start = 0
            for i, nrow_chunk in enumerate(chunk_list):
                logger.info(f'spw {spw}: start reading chunk {i}')

                data_in = dict(
                    (name, tb.getcol(name, chunk_start, nrow_chunk)) for name in self.columns
                )

                yield chunk_start, nrow_chunk, data_in

                logger.info(f'spw {spw}: done reading chunk {i}')
                chunk_start += nrow_chunk

    def _get_chunk_updater(self, spw):
        if spw in self.target_spws:
            logger.info(f'spw {spw}: update chunk')
            return ChunkInterpolator(self.freq_in[spw], self.freq_out[spw])
        else:
            logger.info(f'spw {spw}: leave input chunk as it is')
            return lambda x: x

    def read(self):
        # here, copy input MS to temporary MS
        copy_table_structure(self.vis, self.tmp_vis)
        copy_subtable_rows(self.vis, self.tmp_vis)
        copy_main_columns(self.vis, self.tmp_vis, ignore=self.columns)

        # create generators for lazy read
        self.read_generators = [self._read_main(spw) for spw in self.all_spws]

    def update(self):
        self.update_generators = [
            map(self._get_chunk_updater(spw), chunk) for spw, chunk in enumerate(self.read_generators)
        ]

    def flush(self):
        # flush to the disk
        for spw, update_gen in enumerate(self.update_generators):
            logger.debug(f'spw {spw}: generator {update_gen}')
            taql = self.taql(spw)
            with sdutil.table_selector(self.tmp_vis, taql, nomodify=False) as tb:
                i = 0
                for chunk_start, nrow_chunk, data_out in update_gen:
                    logger.info(f'spw {spw}: start writing chunk {i}')
                    for column, chunk in data_out.items():
                        tb.putcol(column, chunk, chunk_start, nrow_chunk)

                    logger.info(f'spw {spw}: done writing chunk {i}')
                    i += 1

        # finalization
        try:
            rename_table(self.vis, self.backup_vis)
            rename_table(self.tmp_vis, self.vis)
        except Exception as e:
            if os.path.exists(self.tmp_vis):
                shutil.rmtree(self.tmp_vis)
            logger.error(f'Error during renaming. Resulting MS might be corrupted.')
            raise e
        finally:
            if os.path.exists(self.backup_vis):
                shutil.rmtree(self.backup_vis)


class WSUChannelExpander:
    def __init__(self, vis: str, chan_factor: float):
        self.vis = vis
        self.chan_factor = chan_factor

        # get science spws/ddids
        msmd = msmetadata()

        # pick up full resolution science spws
        msmd.open(vis)
        self.science_spws = [int(s) for s in msmd.spwsforintent('OBSERVE_TARGET#ON_SOURCE') if msmd.nchan(s) > 4]
        self.atm_spws = [int(s) for s in msmd.spwsforintent('CALIBRATE_ATMOSPHERE*') if msmd.nchan(s) > 4]
        msmd.close()

        self.target_spws = self.science_spws + self.atm_spws

    def expand(self):
        # process SPECTRAL_WINDOW table
        spw_updater = SpectralWindowUpdater(self.vis, self.target_spws, self.chan_factor)
        spw_updater.read()
        spw_updater.update()
        spw_updater.flush()

        cf_in = spw_updater.get_chan_freq_in()
        cf_out = spw_updater.get_chan_freq_out()

        del spw_updater

        # process SYSCAL table
        # - depends on SPECTRAL_WINDOW information
        syscal_updater = SyscalUpdator(self.vis, self.atm_spws, cf_in, cf_out)
        syscal_updater.read()
        syscal_updater.update()
        syscal_updater.flush()

        del syscal_updater

        # process MAIN table
        # - depends on SPECTRAL_WINDOW information
        main_updater = MainUpdater(self.vis, self.target_spws, cf_in, cf_out)
        main_updater.read()
        main_updater.update()
        main_updater.flush()
