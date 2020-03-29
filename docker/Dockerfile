#FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
#
#RUN apt-get update && \
#    apt-get install -y libssl-dev git make libffi-dev musl-dev openssl && \
#    pip --no-cache-dir install --upgrade pip
#
#COPY requirements.txt /tmp/neat/requirements.txt
#RUN pip install -r /tmp/neat/requirements.txt
#RUN pip install --upgrade jupyter-client jupyter-core
#RUN jupyter notebook --generate-config && \
#    echo "c.NotebookApp.notebook_dir = '/code/notebooks/'" >> "/root/.jupyter/jupyterfin_notebook_config.py"

#WORKDIR /code/notebooks/
#RUN cd /code/notebooks/
#CMD ["/bin/bash", "-c", "/workspace/docker/scripts/run_jupyter_lab.sh"]



#############################################3#
#

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



RUN conda update -n base conda && \
    conda install -c pytorch pytorch=1.3.1 && \
    conda install -c pytorch torchvision=0.4.2

COPY requirements.txt /tmp/neat/requirements.txt
RUN pip install -r /tmp/neat/requirements.txt


# R pre-requisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    gfortran \
    gcc && \
    rm -rf /var/lib/apt/lists/*

# Julia dependencies
# install Julia packages in /opt/julia instead of $HOME
#ENV JULIA_DEPOT_PATH=/opt/julia
#ENV JULIA_PKGDIR=/opt/julia
#ENV JULIA_VERSION=1.3.0
#
#RUN mkdir /opt/julia-${JULIA_VERSION} && \
#    cd /tmp && \
#    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
##    echo "926ced5dec5d726ed0d2919e849ff084a320882fb67ab048385849f9483afc47 *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
#    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
#    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz
#RUN ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia

## Show Julia where conda libraries are \
#RUN mkdir /etc/julia && \
#    echo "push!(Libdl.DL_LOAD_PATH, \"$CONDA_DIR/lib\")" >> /etc/julia/juliarc.jl && \
#    # Create JULIA_PKGDIR \
#    mkdir $JULIA_PKGDIR && \
#    chown $NB_USER $JULIA_PKGDIR && \
#    fix-permissions $JULIA_PKGDIR


## R packages including IRKernel which gets installed globally.
#RUN conda install --quiet --yes \
#    'r-base=3.6.1' \
#    'r-caret=6.0*' \
#    'r-crayon=1.3*' \
#    'r-devtools=2.1*' \
#    'r-forecast=8.7*' \
#    'r-hexbin=1.27*' \
#    'r-htmltools=0.3*' \
#    'r-htmlwidgets=1.3*' \
#    'r-irkernel=1.0*' \
#    'r-nycflights13=1.0*' \
#    'r-plyr=1.8*' \
#    'r-randomforest=4.6*' \
#    'r-rcurl=1.95*' \
#    'r-reshape2=1.4*' \
#    'r-rmarkdown=1.14*' \
#    'r-rsqlite=2.1*' \
#    'r-shiny=1.3*' \
#    'r-sparklyr=1.0*' \
#    'r-tidyverse=1.2*' \
#    'rpy2=2.9*' \
#    && \
#    conda clean --all -f -y && \
#    fix-permissions $CONDA_DIR && \
#    fix-permissions /home/$NB_USER

# Add Julia packages. Only add HDF5 if this is not a test-only build since
# it takes roughly half the entire build time of all of the images on Travis
# to add this one package and often causes Travis to timeout.
#
# Install IJulia as jovyan and then move the kernelspec out
# to the system share location. Avoids problems with runtime UID change not
# taking effect properly on the .local folder in the jovyan home dir.
#RUN julia -e 'import Pkg; Pkg.update()' && \
#    (test $TEST_ONLY_BUILD || julia -e 'import Pkg; Pkg.add("HDF5")') && \
#    julia -e "using Pkg; pkg\"add IJulia\"; pkg\"precompile\"" && \
#    # move kernelspec out of home \
#    mv $HOME/.local/share/jupyter/kernels/julia* $CONDA_DIR/share/jupyter/kernels/ && \
#    chmod -R go+rx $CONDA_DIR/share/jupyter && \
#    rm -rf $HOME/.local && \
#    fix-permissions $JULIA_PKGDIR $CONDA_DIR/share/jupyter

#RUN jupyter notebook --generate-config

#RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /root/mykey.key -out /root/mycert.pem -subj "/C=ES/ST=Madrid/L=Madrid/O=Nextail/OU=NextailCerts/CN=etc/emailAddress=infraops@nextail.com"
RUN apt update -y && \
    apt install graphviz -y

USER $NB_UID

#RUN echo "c.NotebookApp.allow_password_change = False" >> /home/jovyan/.jupyter/jupyter_notebook_config
RUN echo "c.NotebookApp.token = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /home/jovyan/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

CMD ["jupyter", "lab"]