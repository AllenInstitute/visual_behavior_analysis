import setuptools

# https://packaging.python.org/discussions/install-requires-vs-requirements/ this is the way i think it should be -Chris

setuptools.setup(
    name="visual-behavior",
    version="0.8.2.dev0",
    author="Allen Institute for Brain Science",
    author_email="marinag@alleninstitute.org, dougo@alleninstitute.org",
    description="analysis package for visual behavior",
    packages=setuptools.find_packages(exclude=['data', 'figures', 'notebooks', 'scripts']),
    install_requires=[
        "matplotlib",
        "plotly",
        "pandas==0.25.3",
        "six",
        "scikit-learn==0.23.1",
        "scipy>=1.0.0",
        "deepdish>=0.3.6",
        "numpy>=1.9.0",
        "python-dateutil",
        "marshmallow==3.0.0rc4",
        "psycopg2-binary",
        "seaborn",
        'zipfile2; python_version < "3.5"',
        'zipfile36; python_version >= "3.5"',
        'opencv-python',
        'pymongo',
        'pyyaml',
        'h5py>=2.7.1',
        'dash',
    ],
    tests_require=[
        "flake8",
        "pytest",
        "pytest-cov",
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
