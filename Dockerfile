# copy from https://github.com/StevenJSCF/OpenVoice/blob/update-docs/docs/DockerFile
# Use the base image of Ubuntu
FROM ubuntu:22.04  
# as builder

ENV PYTHON_VERSION=python3.10

# Update the system and install necessary dependencies
RUN apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo \
    $PYTHON_VERSION \
    $PYTHON_VERSION-dev \
    python3-distutils \
    python3-pip \
    ffmpeg \
    wget \
    curl \
    git \
    aria2 unzip && \
    ln -sf /usr/bin/$PYTHON_VERSION /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

RUN pip install gradio==3.50.2 langid faster-whisper whisper-timestamped unidecode eng-to-ipa pypinyin cn2an && \
    pip cache purge

WORKDIR /data

# v2
RUN wget https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip && \
    unzip checkpoints_v2_0417.zip && \
    rm -f checkpoints_v2_0417.zip

# v1
# RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/OpenVoice/resolve/main/checkpoints_1226.zip -d /data -o checkpoints_1226.zip && \
#     unzip /data/checkpoints_1226.zip && \
#     rm -rf /app/openvoice/checkpoints_1226.zip

# Set the working directory in the container
WORKDIR /app/openvoice

ADD . .

RUN pip install -e . && \
    pip install soundfile librosa inflect jieba silero && \
    pip cache purge && \
    sed -i "s/demo.launch(debug=True, show_api=True, share=args.share)/demo.launch(server_name='0.0.0.0', debug=True, show_api=True, share=args.share)/" ./openvoice/openvoice_app.py && \
    ln -sf /data/checkpoints_v2 ./checkpoints

ENV HF_ENDPOINT=https://hf-mirror.com
ENV LD_PRELOAD=$LD_PRELOAD:/usr/local/lib/$PYTHON_VERSION/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

RUN mkdir -p /root/.cache/torch/hub && \
    curl -sL https://github.com/snakers4/silero-vad/zipball/master -o /root/.cache/torch/hub/master.zip && \
    curl -sL https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_aarch64 -o /usr/local/lib/$PYTHON_VERSION/dist-packages/gradio/frpc_linux_aarch64_v0.2

# v2
RUN pip install git+https://github.com/myshell-ai/MeloTTS.git && \
    pip cache purge && \
    python3 -m unidic download

# FROM  ubuntu:20.04
# ENV  LANGUAGE zh_CN.UTF-8
# ENV  LANG zh_CN.UTF-8
# ENV  LC_ALL zh_CN.UTF-8
# RUN apt-get update && apt-get install -y \
#     locales \
#     libxcb1 libgomp1 && \
#   apt-get autoclean && rm -rf /var/lib/apt/lists/* && \
#   locale-gen zh_CN && \
#   locale-gen zh_CN.UTF-8
  
# WORKDIR /workdir
# COPY --from=builder /app/openvoice/dist/openvoice_app /workdir/openvoice

# ENV HF_ENDPOINT=https://hf-mirror.com
# ENV LD_PRELOAD=$LD_PRELOAD:/usr/local/lib/$PYTHON_VERSION/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

# EXPOSE 7860
# CMD ["./openvoice", "--share"]
# Default command when the container is started
CMD ["python3", "-m", "openvoice.openvoice_app_v2"]

