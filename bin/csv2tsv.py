#!/usr/bin/env python

# https://unix.stackexchange.com/questions/359832/converting-csv-to-tsv

import sys
import csv
from StringIO import StringIO

for line in sys.stdin:
  print('\t'.join(csv.reader(StringIO(line)).next()))
