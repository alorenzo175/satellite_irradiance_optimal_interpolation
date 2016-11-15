import os
from distutils.core import setup
import versioneer

PACKAGE = 'satoi'

SHORT_DESC = 'Tools to perform optimal interpolation of satellite data'
AUTHOR = 'Tony Lorenzo'
MAINTAINER_EMAIL = 'atlorenzo@email.arizona.edu'

setup(
    name=PACKAGE,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=SHORT_DESC,
    author=AUTHOR,
    maintainer_email=MAINTAINER_EMAIL,
    packages={'satoi': 'satoi'},
    scripts=[os.path.join('scripts', s) for s in os.listdir('scripts')])
