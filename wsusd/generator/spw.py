import os

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
            self._duplicate()

    def _expand_spectral_window(self):
        table_name = os.path.join(self.vis, 'SPECTRAL_WINDOW')
        self.extra_spw = []
        with sdutil.table_manager(table_name, nomodify=False) as tb:
            for base_spw in self.target_spws:
                new_spw = tb.nrows()
                logger.info(f'duplicating spw {base_spw}: spw {new_spw} will be added')
                tb.copyrows(table_name, base_spw, nrow=1)
                self.extra_spw.append(new_spw)

    def _expand_data_description(self):
        table_name = os.path.join(self.vis, 'DATA_DESCRIPTION')
        self.extra_dd = []
        with sdutil.table_manager(table_name, nomodify=False) as tb:
            for base_spw, new_spw in zip(self.target_spws, self.extra_spw):
                base_dd = self.spw_dd_map[base_spw]
                new_dd = tb.nrows()
                logger.info(f'duplicating dd {base_dd} (spw {base_spw}): '
                            f'dd {new_dd} (spw {new_spw}) will be added')
                tb.copyrows(table_name, base_dd, nrow=1)
                tb.putcell('SPECTRAL_WINDOW_ID', new_dd, new_spw)
                self.extra_dd.append(new_dd)

    def _expand_syscal(self):
        pass

    def _expand_main(self):
        pass

    def _duplicate(self):
        # duplicate science & atm spws
        logger.info('_duplicate')

        # process SPECTRAL_WINDOW
        self._expand_spectral_window()

        # process DATA_DESCRIPTION
        self._expand_data_description()

        # process SYSCAL
        self._expand_syscal()

        # process MAIN
        self._expand_main()
