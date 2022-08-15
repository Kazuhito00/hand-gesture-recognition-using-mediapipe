FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        python3-pip \
        python3-mock \
        libpython3-all-dev \
        python-is-python3 \
        pkg-config \
        software-properties-common \
        nano \
        sudo \
        libgl1-mesa-dev \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pip -U \
    && pip install -U onnx \
    && pip install -U onnx-simplifier \
    && python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
    && pip install -U simple_onnx_processing_tools \
    && pip install tensorflow==2.8.0 \
    && pip install mediapipe==0.8.1 --no-deps \
    && pip install opencv-python==4.1.2.30 \
    && pip install scikit-learn==0.23.2 \
    && pip install matplotlib==3.3.2

ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/home/${USERNAME}/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}
RUN echo 'export QT_X11_NO_MITSHM=1' >> ${HOME}/.bashrc
RUN echo 'sudo chmod 776 /dev/video*' >> ${HOME}/.bashrc
