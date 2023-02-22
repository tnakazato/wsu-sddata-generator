import os

from casatasks.private import sdutil

from _logging import get_logger
from generator.util import get_target_spws

logger = get_logger(__name__)


class WSUSpwExpander:
    def __init__(self, vis: str, spw_factor: int):
        self.vis = vis
        self.spw_factor = spw_factor

        self.science_spws, self.atm_spws = get_target_spws(self.vis)
        self.target_spws = sorted(self.science_spws + self.atm_spws)
        logger.info(f'target spws: {self.target_spws}')

    def expand(self):
        num_duplication = self.spw_factor - 1
        logger.info(f'duplicate {num_duplication} times')
        for i in range(num_duplication):
            self._duplicate()

    def _expand_spectral_window(self):
        table_name = os.path.join(self.vis, 'SPECTRAL_WINDOW')
        with sdutil.table_manager(table_name, nomodify=False) as tb:
            for spw in self.target_spws:
                logger.info(f'duplicating spw {spw}: spw {tb.nrows()} will be added')
                tb.copyrows(table_name, spw, nrow=1)

    def _expand_data_description(self):
        pass

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
