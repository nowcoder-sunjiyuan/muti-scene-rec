FROM tensorflow/tensorflow:2.16.1-gpu-jupyter
LABEL authors="sun"

COPY ./requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple