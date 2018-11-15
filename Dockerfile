FROM continuumio/anaconda:latest

# Install environment dependencies:
RUN apt-get update -yqq && apt-get install -yqq --no-install-recommends xvfb emacs curl apt-utils wget libgl1-mesa-glx && apt-get -q clean

# Install python dependencies:
RUN conda update -n base conda
RUN conda install -y scipy numpy pandas scikit-learn subprocess32 cython

# Set up build:
RUN mkdir -p /data
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
