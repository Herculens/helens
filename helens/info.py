# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2024, helens developers and contributors

# Set the package release version
version_info = (0, 0, 1)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__author__ = 'Austin Peel, Aymeric Galan'
__email__ = 'aymeric.galan@gmail.com'
__year__ = '2024'
__url__ = 'https://github.com/Herculens/helens'
__description__ = 'Lens equation solver for strong lensing applications, written in JAX.'
__python__ = '>=3.7'
__requires__ = [
    'jax>=0.3.14', 
    'jaxlib>=0.3.14', 
]  # Package dependencies

# Default package properties
__license__ = 'MIT'
__about__ = ('{} Author: {}, Email: {}, Year: {}, {}'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pep8']
