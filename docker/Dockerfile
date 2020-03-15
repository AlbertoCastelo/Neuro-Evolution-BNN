FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN apt-get update && \
    apt-get install -y libssl-dev git make libffi-dev musl-dev openssl && \
    pip --no-cache-dir install --upgrade pip

COPY requirements.txt /tmp/neat/requirements.txt
RUN pip install -r /tmp/neat/requirements.txt
RUN pip install --upgrade jupyter-client jupyter-core
#RUN jupyter notebook --generate-config && \
#    echo "c.NotebookApp.notebook_dir = '/code/notebooks/'" >> "/root/.jupyter/jupyterfin_notebook_config.py"

#WORKDIR /code/notebooks/
#RUN cd /code/notebooks/
CMD ["/bin/bash", "-c", "/workspace/docker/scripts/run_jupyter_lab.sh"]
