import os

from casatasks import importasdm

from _logging import get_logger

logger = get_logger(__name__)


def run_importasdm(asdm, vis):
    logger.info(f'Generating MS {vis}')
    basename, _ = os.path.splitext(vis)
    try:
        importasdm(
            asdm=asdm, vis=vis,
            createmms=False, ocorr_mode='ao', lazy=False,
            asis='SBSummary ExecBlock Annotation Antenna Station Receiver Source CalAtmosphere CalWVR SpectralWindow',
            process_caldevice=False, savecmds=True,
            outfile=f'{basename}.flagonline.txt', overwrite=False,
            bdfflags=True, with_pointing_correction=True
        )
    except Exception as e:
        if 'You have specified an existing MS' in str(e):
            pass
        else:
            raise e

    return vis


