import os

import numpy as np

from casatasks.private import sdutil

from _logging import get_logger
from generator.util import get_spw_dd_map, get_target_spws

logger = get_logger(__name__)


class WSUSpwExpander:
    def __init__(self, vis: str, spw_factor: int):
        self.vis = vis
        self.spw_factor = spw_factor

        self.science_spws, self.atm_spws = get_target_spws(self.vis)
        self.target_spws = sorted(set(self.science_spws + self.atm_spws))
        logger.info(f'target spws: {self.target_spws}')

        self.spw_dd_map = get_spw_dd_map(self.vis)

    def expand(self):
        num_duplication = self.spw_factor - 1
        logger.info(f'duplicate {num_duplication} times')
        for i in range(num_duplication):
            logger.info(f'start duplication cycle {i}')
            self._duplicate()
            logger.info(f'done duplication cycle {i}')

    def _expand_spectral_window(self):
        table_name = os.path.join(self.vis, 'SPECTRAL_WINDOW')
        self.extra_spw = []
        with sdutil.table_manager(table_name, nomodify=False) as tb:
            for base_spw in self.target_spws:
                new_spw = tb.nrows()
                logger.info(f'duplicating spw {base_spw}: spw {new_spw} will be added')
                tb.copyrows(table_name, base_spw, nrow=1)
                self.extra_spw.append(new_spw)

    def __copy_selected_rows(self, tb, taql):
        table_name = tb.name()

        selected = tb.query(taql)
        try:
            if selected.nrows() == 0:
                return

            selected.copyrows(table_name)
        finally:
            selected.close()

    def __expand_subtable(self, subtable_name):
        full_table_name = os.path.join(self.vis, subtable_name)
        with sdutil.table_manager(full_table_name, nomodify=False) as tb:
            nrow_before = tb.nrows()

            for base_spw, new_spw in zip(self.target_spws, self.extra_spw):
                logger.info(f'duplicating {subtable_name} rows for spw {base_spw}: '
                            f'spw {new_spw} will be assigned')

                startrow = tb.nrows()
                taql = f'SPECTRAL_WINDOW_ID == {base_spw}'
                self.__copy_selected_rows(tb, taql)
                nrow = tb.nrows() - startrow
                spwcol = np.zeros(nrow, dtype=int) + new_spw
                tb.putcol('SPECTRAL_WINDOW_ID', spwcol, startrow, nrow)

            nrow_after = tb.nrows()

        return nrow_before, nrow_after

    def _expand_data_description(self):
        nrow_before, nrow_after = self.__expand_subtable('DATA_DESCRIPTION')
        self.extra_dd = list(range(nrow_before, nrow_after))
        assert len(self.extra_dd) == len(self.extra_spw)

    def _expand_syscal(self):
        self.__expand_subtable('SYSCAL')

    def _expand_feed(self):
        self.__expand_subtable('FEED')

    def _expand_source(self):
        self.__expand_subtable('SOURCE')

    def _expand_main(self):
        table_name = self.vis
        with sdutil.table_manager(table_name, nomodify=False) as tb:
            for base_spw, new_spw, new_dd in zip(self.target_spws, self.extra_spw, self.extra_dd):
                # here we assume spw-dd mapping is one-to-one
                # (mapping is one-to-many in general)
                base_dd = self.spw_dd_map[base_spw][0]
                logger.info(f'duplicating MAIN rows for dd {base_dd} (spw {base_spw}): '
                            f'dd {new_dd} (spw {new_spw}) will be assigned')

                startrow = tb.nrows()
                taql = f'DATA_DESC_ID == {base_dd}'
                self.__copy_selected_rows(tb, taql)
                nrow = tb.nrows() - startrow
                ddcol = np.zeros(nrow, dtype=int) + new_dd
                tb.putcol('DATA_DESC_ID', ddcol, startrow, nrow)

    def _duplicate(self):
        # duplicate science & atm spws
        # process SPECTRAL_WINDOW
        self._expand_spectral_window()

        # process DATA_DESCRIPTION
        self._expand_data_description()

        # process SYSCAL
        self._expand_syscal()

        # process FEED
        self._expand_feed()

        # process SOURCE
        self._expand_source()

        # process MAIN
        self._expand_main()
