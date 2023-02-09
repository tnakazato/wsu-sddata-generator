# WSU-SD Data Generator

Data Generator for ALMA-WSU SD.

```
usage: wsusd [-h] [--version] [--debug] [--backup-ms] asdm_name [chan_factor]

From the input data, generates single dish artificial data that emulates ALMA-
WSU observation.

positional arguments:
  asdm_name        Path to the ASDM on disk.
  chan_factor      Channel expantion factor. For example, setting this to 10
                   will produce the data with 10 times more channels than
                   input data. Default is 10.

optional arguments:
  -h, --help       show this help message and exit
  --version        show program's version number and exit
  --debug, -d      Debug mode.
  --backup-ms, -b  Back up MS before manipulating.
```

