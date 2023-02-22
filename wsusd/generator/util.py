from casatools import msmetadata

from typing import List, Tuple


def get_target_spws(vis: str) -> Tuple[List[int], List[int]]:
    # get science spws/ddids
    # pick up full resolution science spws
    msmd = msmetadata()
    msmd.open(vis)
    science_spws = [int(s) for s in msmd.spwsforintent('OBSERVE_TARGET#ON_SOURCE') if msmd.nchan(s) > 4]
    atm_spws = [int(s) for s in msmd.spwsforintent('CALIBRATE_ATMOSPHERE*') if msmd.nchan(s) > 4]
    msmd.close()

    return science_spws, atm_spws


def get_spw_dd_map(vis: str) -> dict:
    # get spw->ddid mapping
    msmd = msmetadata()
    msmd.open(vis)
    num_spws = msmd.nspw()
    spw_ddid_map = dict((spw, msmd.datadescids(spw)) for spw in range(num_spws))
    msmd.close()

    return spw_ddid_map
