ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Set when building on Travis so that certain long-running build steps can
# be skipped to shorten build time.
ARG TEST_ONLY_BUILD

USER root

ADD docker/ssh/secrets /root/.ssh/id_rsa

RUN chmod 600 /root/.ssh/id_rsa \
    && echo "[url \"git@github.com:\"]\n\tinsteadOf = https://github.com/" >> /root/.gitconfig \
    && echo "StrictHostKeyChecking no " > /root/.ssh/config

RUN apt update -y && \
    apt install graphviz -y


RUN conda update -n base conda && \
    conda install -c pytorch pytorch=1.3.1 && \
    conda install -c pytorch torchvision=0.4.2

COPY requirements.txt /tmp/neat/requirements.txt
RUN pip install -r /tmp/neat/requirements.txt


COPY docker/scripts/install-final-dependencies.sh /tmp/scripts/install-final-dependencies.sh
RUN bash /tmp/scripts/install-final-dependencies.sh


RUN echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py

EXPOSE 8888
