import argparse
import os
import shutil

from wsusd._logging import get_logger, set_debug_level
from wsusd.asdm2ms import is_asdm, run_importasdm
from wsusd.generator.channel import WSUChannelExpander
from wsusd.generator.spw import WSUSpwExpander
from wsusd.version import VERSION

DESCRIPTION = '''
From the input data, generates single dish artificial data that emulates
ALMA-WSU observation.'''

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {VERSION}'
    )
    parser.add_argument(
        '--debug', '-d',
        help='Debug mode.',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--dry-run',
        help='Dry-run mode. Just print input parameters and exit.',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--backup-ms', '-b',
        help='Back up MS before manipulating.',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--chan-factor', '-c',
        help='Channel expansion factor. '
             'For example, setting this to 10 will produce the data '
             'with 10 times more channels than input data. '
             'Default is 1 (no channel expansion).',
        type=float,
        action='store',
        default=1
    )
    parser.add_argument(
        '--spw-factor', '-s',
        help='Spectral Window (spw) expansion factor. '
             'For example, setting this to 2 will duplicate data for science '
             'spws twice. Default is 1 (no spw expansion).',
        type=int,
        action='store',
        default=1
    )
    parser.add_argument(
        'asdm_name',
        help='Path to the ASDM on disk.',
        type=str,
        nargs=1,
    )
    args = parser.parse_args()

    return args


def generate_ms_name(asdm, chan_factor, spw_factor):
    logger.debug('%s, %s' % (asdm, type(asdm)))
    asdm_basename = os.path.basename(asdm.rstrip('/'))
    basename = asdm_basename

    if chan_factor > 1:
        basename += f'.{chan_factor}xchan'

    if spw_factor > 1:
        basename += f'.{spw_factor}xspw'

    vis = f'{basename}.ms'

    return vis


def generate(asdm, chan_factor, spw_factor, backup_ms=False, dry_run=False):
    # run importasdm
    logger.info('Input Parameter:')
    logger.info(f'  asdm = "{asdm}"')
    logger.info(f'  chan_factor = {chan_factor}')
    logger.info(f'  spw_factor = {spw_factor}')
    logger.info(f'  backup_ms = {backup_ms}')

    if not chan_factor > 0:
        msg = f'Invalid chan_factor was given ({chan_factor}). Must be > 0.'
        logger.error(msg)
        raise ValueError(msg)

    if is_asdm(asdm):
        if dry_run:
            return

        logger.info('Running importasdm task to generate MS')
        vis = generate_ms_name(asdm, chan_factor, spw_factor)
        run_importasdm(asdm, vis)
        logger.info(f'Created {vis}')
    else:
        # assuming input data is MS
        vis = asdm

    if backup_ms:
        vis_bak = f'{vis}.bak'
        logger.info(
            f'Back up MS before manipulation. Back up file is {vis_bak}'
        )
        shutil.copytree(vis, vis_bak)

    if chan_factor > 1:
        logger.info('Updating channel structure')
        generator = WSUChannelExpander(vis, chan_factor)
        generator.expand(dry_run=dry_run)

    if spw_factor > 1:
        logger.info('Updating spw structure')
        generator = WSUSpwExpander(vis, spw_factor)
        generator.expand(dry_run=dry_run)

    logger.info(f'Completed{" (dry run)" if dry_run else ""}: '
                f'Name of the output MS is {vis}')


def main():
    # parse user inputs
    args = parse_args()

    if args.dry_run:
        print(args)

    if args.debug:
        set_debug_level(logger)

    logger.debug(args)

    asdm = args.asdm_name[0]
    chan_factor = args.chan_factor
    spw_factor = args.spw_factor
    backup_ms = args.backup_ms
    dry_run = args.dry_run

    # expand nchan
    generate(asdm, chan_factor, spw_factor, backup_ms, dry_run)


if __name__ == '__main__':
    main()
