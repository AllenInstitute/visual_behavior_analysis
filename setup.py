import setuptools

# https://packaging.python.org/discussions/install-requires-vs-requirements/ this is the way i think it should be -Chris

def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements

setuptools.setup(
    name="visual-behavior",
    version="0.6.0.dev0",
    author="Nicholas Cain, Marina Garrett, Justin Kiggins, Chris Mochizuki, Doug Ollerenshaw, Nick Ponvert, and many others",
    author_email="nicholasc, marinag, chrism, dougo, nick.ponvert <user>@alleninstitute.org",
    description="analysis package for visual behavior",
    packages=setuptools.find_packages(exclude=['data', 'figures', 'notebooks', 'scripts']),
    package_data={'': ['requirements.txt']},
    install_requires=get_requirements()[1:], #ignore the first line which has --extra-index-url https://aibs.jfrog.io/aibs/api/pypi/ni-pypi-local/simple
    dependency_links=['https://aibs.jfrog.io/aibs/api/pypi/ni-pypi-local/simple/allensdk'],
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
