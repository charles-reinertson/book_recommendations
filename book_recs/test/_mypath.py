import os
import sys


thisdir = os.path.dirname(__file__)
libdir = os.path.join(thisdir, '../../')


if libdir not in sys.path:
    sys.path.insert(0, libdir)
    sys.path.insert(0, libdir + 'book_recs/')

os.chdir(libdir)

