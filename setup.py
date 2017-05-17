import setuptools

setuptools.setup(
    name="braintv_behav",
    version="0.1.0",
    url="http://stash.corp.alleninstitute.org/users/justink/repos/braintv_behavior_piloting/browse",

    author="Justin Kiggins",
    author_email="justink@alleninstitute.org",

    description="project for analyzing pilot behavior for BrainTV",
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(exclude=['data','figures','notebooks','scripts']),

    # install_requires=[
    #     'Click',
    # ],

    # entry_points='''
    #     [console_scripts]
    #     summary_csv=braintv_pilot.summary:load_and_save
    # ''',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
    ],
)
