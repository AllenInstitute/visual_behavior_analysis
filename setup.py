import setuptools

# https://packaging.python.org/discussions/install-requires-vs-requirements/ this is the way i think it should be -Chris

setuptools.setup(
    name="visual-behavior",
    version="0.4.3",
    author="Justin Kiggins",
    author_email="justink@alleninstitute.org",
    description="analysis package for visual behavior",
    packages=setuptools.find_packages(exclude=['data', 'figures', 'notebooks', 'scripts']),
    install_requires=[
        "matplotlib",
        "pandas",
        "six",
        "scikit-learn>=0.19.2",
        "scipy>=1.0.0",
        "deepdish>=0.3.6",
        "numpy>=1.9.0",  # for science some packages need to be pinned
        "python-dateutil",
        "marshmallow==3.0.0b11",
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
