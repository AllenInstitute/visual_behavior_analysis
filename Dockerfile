FROM continuumio/anaconda:latest

# Install environment dependencies:
RUN apt-get update -yqq && apt-get install -yqq --no-install-recommends xvfb emacs curl apt-utils wget libgl1-mesa-glx && apt-get -q clean

# Install python dependencies:
RUN conda update -n base conda
RUN conda install -y scipy numpy pandas scikit-learn subprocess32 cython entrypoints

# Set up build:
RUN mkdir -p /data
RUN mkdir -p /allen/aibs/informatics/swdb2018/visual_behavior/702134928_363887_180524_VISal_175_Vip_2P6_behavior_sessionC
RUN mkdir -p /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508
RUN mkdir -p /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_800402935
RUN mkdir -p /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix

ARG PKG
RUN mkdir -p /${PKG}
WORKDIR /${PKG}

RUN pip install --upgrade pip
RUN pip install virtualenv
COPY requirements_dev.txt requirements_dev.txt
COPY requirements.txt requirements.txt
COPY tox.ini tox.ini
COPY setup.py setup.py
COPY test.sh test.sh
COPY build_bdist_wheel.sh build_bdist_wheel.sh
COPY ${PKG} ${PKG}
COPY tests tests

RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt
RUN pip install .

RUN rm -rf tests/__pycache__
RUN find . -name \*.pyc -delete

# Copy over regression test fixture files:
COPY fixtures/181119092559_412629_a3775e3e-e1ca-474a-b413-91cccd6d886f.pkl /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119092559_412629_a3775e3e-e1ca-474a-b413-91cccd6d886f.pkl
COPY fixtures/181119102010_421137_c108dc71-ef5e-46ad-8d85-8da0fdaf7d3d.pkl /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119102010_421137_c108dc71-ef5e-46ad-8d85-8da0fdaf7d3d.pkl
COPY fixtures/181119150503_416656_2b0893fe-843d-495e-bceb-83b13f2b02dc.pkl /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119150503_416656_2b0893fe-843d-495e-bceb-83b13f2b02dc.pkl
COPY fixtures/181119135416_424460_b6daf247-2caf-4f38-9eb1-ab97825923cd.pkl /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119135416_424460_b6daf247-2caf-4f38-9eb1-ab97825923cd.pkl
COPY fixtures/778113069_stim.pkl /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/778113069_stim.pkl
COPY fixtures/181119134201_402329_b75a87d0-8178-4171-a3b2-7cea3ae8e118.pkl /allen/programs/braintv/workgroups/nc-ophys/visual_behavior/test_fixtures/181119134201_402329_b75a87d0-8178-4171-a3b2-7cea3ae8e118.pkl

COPY fixtures/nimages_0_20170714.zip /allen/aibs/mpe/Software/stimulus_files/nimages_0_20170714.zip
COPY fixtures/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl /allen/programs/braintv/workgroups/nc-ophys/Doug/Stimulus_Code/image_dictionaries/Natural_Images_Lum_Matched_set_training_2017.07.14.pkl

COPY fixtures/702134928_363887_180524_VISal_175_Vip_2P6_behavior_sessionC /allen/aibs/informatics/swdb2018/visual_behavior/702134928_363887_180524_VISal_175_Vip_2P6_behavior_sessionC
RUN chmod -R a-w /allen/aibs/informatics/swdb2018/visual_behavior/702134928_363887_180524_VISal_175_Vip_2P6_behavior_sessionC

# Ophys test assets: 
COPY fixtures/702013508_363887_20180524142941_sync.h5 /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508
COPY fixtures/702013508_363887_20180524142941_stim.pkl  /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508
COPY fixtures/objectlist.txt /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_800402935 
COPY fixtures/702134928_input_extract_traces.json /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed
COPY fixtures/702134928_dff.h5 /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928
COPY fixtures/702134928_rigid_motion_transform.csv /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed 
COPY fixtures/maxInt_a13a.png /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_800402935 
COPY fixtures/702134928_demixed_traces.h5 /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/demix 
COPY fixtures/avgInt_a1X.png /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed/ophys_cell_segmentation_run_800402935 
COPY fixtures/roi_traces.h5 /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/ophys_experiment_702134928/processed 

CMD ["/bin/bash"]


# COPY ./.pypirc /root/.pypirc
# COPY ./requirements_dev.txt /edf/requirements_dev.txt
# COPY ./requirements.txt /edf/requirements.txt
# COPY ./edf /edf/edf

# COPY ./test.sh /edf/test.sh

# RUN pip install -r /edf/requirements_dev.txt
# RUN ping -c3 ibs-afserver-vm1.corp.alleninstitute.org
# RUN ping -c3 10.128.108.53
# RUN ping -c3 stash.corp.alleninstitute.org
# RUN ping -c3 aibs-artifactory
# RUN curl -o visual_behavior-0.3.0-py2-none-any.whl http://aibs-artifactory/artifactory/api/pypi/pypi-local/visual_behavior/0.3.0/visual_behavior-0.3.0-py2-none-any.whl
# RUN pip install http://aibs-artifactory/artifactory/api/pypi/pypi-local/visual_behavior/0.3.0/visual_behavior-0.3.0-py2-none-any.whl
# RUN pip install --user http://aibs-artifactory/artifactory/api/pypi/pypi-local/visual_behavior/0.3.0/visual_behavior-0.3.0-py2-none-any.whl
# RUN pip install -i http://aibs-artifactory/artifactory/api/pypi/pypi-local  --trusted-host aibs-artifactory -r /edf/requirements.txt
# RUN pip install -r /edf/requirements.txt
# RUN pip search -i http://aibs-artifactory/artifactory/api/pypi/pypi-local visual_behavior
# RUN pip install --extra-index-url https://pypi.org/simple -i http://aibs-artifactory/artifactory/api/pypi/pypi-local/simple  --trusted-host aibs-artifactory visual_behavior==0.3.0

# RUN pip install visual_behavior==0.3.0
# RUN pip install -r /edf

# WORKDIR /edf

# CMD [ "/edf/test.sh" ]

# # Create the group and user to match the VM:
# RUN mkdir -p /home/mtrain/data 
# RUN mkdir -p /home/mtrain/app/mtrain_api
# RUN mkdir -p /home/mtrain/app/mtrain_api/mtrain_api/static

# # set working directory

# WORKDIR /home/mtrain/app

# # add requirements
# COPY --chown=mtrain:mtrain ./mtrain/ /home/mtrain/app/mtrain/
# COPY --chown=mtrain:mtrain ./tests/ /home/mtrain/app/mtrain_api/tests/
# COPY --chown=mtrain:mtrain ./mtrain_api/ /home/mtrain/app/mtrain_api/mtrain_api/
# COPY --chown=mtrain:mtrain ./requirements.txt /home/mtrain/app/mtrain_api/requirements.txt
# COPY --chown=mtrain:mtrain ./requirements /home/mtrain/app/mtrain_api/requirements

# RUN pip install --user /home/mtrain/app/mtrain
# RUN pip install --user -r /home/mtrain/app/mtrain_api/requirements.txt
# RUN pip install --user -r /home/mtrain/app/mtrain_api/requirements/dev.txt
# RUN pip install --user http://aibs-artifactory/artifactory/api/pypi/pypi-local/visual_behavior/0.3.0/visual_behavior-0.3.0-py2-none-any.whl


# # add entrypoint.sh
# COPY --chown=mtrain:mtrain ./test.sh /home/mtrain/app/mtrain_api/test.sh
# COPY --chown=mtrain:mtrain ./entrypoint.sh /home/mtrain/app/mtrain_api/entrypoint.sh
# COPY --chown=mtrain:mtrain ./production.sh /home/mtrain/app/mtrain_api/production.sh
# COPY --chown=mtrain:mtrain ./autoapp.py /home/mtrain/app/mtrain_api/autoapp.py
# COPY --chown=mtrain:mtrain ./package.json /home/mtrain/app/mtrain_api/package.json
# COPY --chown=mtrain:mtrain ./webpack.config.js /home/mtrain/app/mtrain_api/webpack.config.js
# COPY --chown=mtrain:mtrain ./assets /home/mtrain/app/mtrain_api/assets/


# # ENV PYTHONPATH /mtrain/app
# ENV PATH="/home/mtrain/.local/bin:${PATH}"
# ENV WEBPACK_MANIFEST_PATH='/home/mtrain/app/mtrain_api/mtrain_api/webpack/manifest.json'
# ENV MTRAIN_API_STATIC_FOLDER='/home/mtrain/app/mtrain_api/mtrain_api/static'
# ENV MTRAIN_API_TEMPLATE_FOLDER='/home/mtrain/app/mtrain_api/mtrain_api/templates'

# WORKDIR /home/mtrain/app/mtrain_api

# RUN rm -rf tests/__pycache__
# RUN find . -name \*.pyc -delete
# RUN npm install
# RUN npm run build

# # run server
# CMD ["/home/mtrain/app/mtrain_api/entrypoint.sh"]
