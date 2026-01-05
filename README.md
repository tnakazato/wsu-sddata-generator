# WSU-SD Data Generator

Data Generator for ALMA-WSU SD.

## Installation

```
pip3 install [-e] .
```

## Usage

```
usage: wsusd [-h] [--version] [--debug] [--dry-run] [--backup-ms]
             [--chan-factor CHAN_FACTOR] [--spw-factor SPW_FACTOR]
             asdm_name

From the input data, generates single dish artificial data that emulates ALMA-
WSU observation.

positional arguments:
  asdm_name             Path to the ASDM on disk.

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --debug, -d           Debug mode.
  --dry-run             Dry-run mode. Just print input parameters and exit.
  --backup-ms, -b       Back up MS before manipulating.
  --chan-factor CHAN_FACTOR, -c CHAN_FACTOR
                        Channel expansion factor. For example, setting this to
                        10 will produce the data with 10 times more channels
                        than input data. Default is 1 (no channel expansion).
  --spw-factor SPW_FACTOR, -s SPW_FACTOR
                        Spectral Window (spw) expansion factor. For example,
                        setting this to 2 will duplicate data for science spws
                        twice. Default is 1 (no spw expansion).
```

