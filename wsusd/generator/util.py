from casatools import msmetadata

from typing import List, Tuple

import numpy as np

from wsusd._logging import get_logger

logger = get_logger(__name__)


def get_target_spws(vis: str) -> Tuple[List[int], List[int]]:
    # get science spws/ddids
    # pick up full resolution science spws
    msmd = msmetadata()
    msmd.open(vis)
    science_spws = [
        int(s) for s in msmd.spwsforintent('OBSERVE_TARGET#ON_SOURCE')
        if msmd.nchan(s) > 4
    ]
    atm_spws = [
        int(s) for s in msmd.spwsforintent('CALIBRATE_ATMOSPHERE*')
        if msmd.nchan(s) > 4
    ]
    msmd.close()

    return science_spws, atm_spws


def get_spw_dd_map(vis: str) -> dict:
    # get spw->ddid mapping
    msmd = msmetadata()
    msmd.open(vis)
    num_spws = msmd.nspw()
    spw_ddid_map = dict(
        (spw, msmd.datadescids(spw)) for spw in range(num_spws)
    )
    msmd.close()

    return spw_ddid_map


def copy_selected_main_rows(tb, taql):
    table_name = tb.name()

    selected = tb.query(taql)
    try:
        nrows = selected.nrows()

        if nrows == 0:
            return

        colnames = tb.colnames()
        start_row = tb.nrows()
        mandatory_scalar_cols = [
            "TIME", "ANTENNA1", "ANTENNA2", "FEED1", "FEED2",
            "DATA_DESC_ID", "PROCESSOR_ID", "FIELD_ID",
            "INTERVAL", "EXPOSURE", "TIME_CENTROID",
            "SCAN_NUMBER", "ARRAY_ID", "OBSERVATION_ID",
            "STATE_ID", "FLAG_ROW"
        ]
        mandatory_fixed_array_cols = [
            "UVW", "SIGMA", "WEIGHT"
        ]
        tb.addrows(nrows)
        cols = mandatory_scalar_cols + mandatory_fixed_array_cols
        for col in cols:
            logger.info(f"copying {col}")
            col_data = selected.getcol(col)
            tb.putcol(col, col_data, startrow=start_row, nrow=nrows)
            logger.info(f"done copying {col}")

        # data columns and flag column
        data_flag_cols = list(set(colnames).intersection(["DATA", "CORRECTED_DATA", "FLOAT_DATA", "FLAG"]))
        dminfo = tb.getdminfo()
        for col in data_flag_cols:
            logger.info(f"copyting {col}")
            col_dminfo = next((v for v in dminfo.values() if col in v["COLUMNS"]))
            if col_dminfo["TYPE"] == "TiledShapeStMan":
                tile_shape = col_dminfo["SPEC"]["HYPERCUBES"]["*1"]["TileShape"]
                cube_shape = col_dminfo["SPEC"]["HYPERCUBES"]["*1"]["CubeShape"]
                chunk_row_size = max(int(10e9 / (np.prod(cube_shape[:2]) * 128)), tile_shape[2])
                logger.info(f"chunk_row_size for {col} is set to {chunk_row_size}")
                num_chunk = int(np.ceil(nrows / chunk_row_size))
                for i in range(num_chunk):
                    start = i * chunk_row_size
                    end = min((i + 1) * chunk_row_size, nrows)
                    logger.info(f"copying {col}: rows {start}-{end}/{nrows}")
                    col_data = selected.getcol(col, startrow=start, nrow=end - start)
                    tb.putcol(col, col_data, startrow=start_row + start, nrow=end - start)
                    logger.info(f"done copying {col}: rows {start}-{end}/{nrows}")
            else:
                for i in range(nrows):
                    if selected.iscelldefined(col, i):
                        if i % 100 == 0:
                            logger.info(f"copying {col}: row {i}/{nrows}")
                        cell_data = selected.getcell(col, i)
                        tb.putcell(col, start_row + i, cell_data)
                        if i % 100 == 0:
                            logger.info(f"done copying {col}: row {i}/{nrows}")
            logger.info(f"done copying {col}")
        other_cols = set(colnames).difference(cols + data_flag_cols)
        for col in other_cols:
            logger.info(f"copying {col}")
            for i in range(nrows):
                if selected.iscelldefined(col, i):
                    if i % 100 == 0:
                        logger.info(f"copying {col}: row {i}/{nrows}")
                    cell_data = selected.getcell(col, i)
                    tb.putcell(col, start_row + i, cell_data)
                    if i % 100 == 0:
                        logger.info(f"done copying {col}: row {i}/{nrows}")
            logger.info(f"done copying {col}")
    finally:
        selected.close()
