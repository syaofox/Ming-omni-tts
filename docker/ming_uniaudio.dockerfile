FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get clean && \
    apt-get update

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        vim \
        libsndfile1 \
        build-essential \
        python3-dev \
        nodejs \
        npm \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3.11 /usr/bin/pip

RUN pip install --no-cache-dir --upgrade pip
RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple/

RUN pip install --no-cache-dir \
    tokenizers \
    # grouped_gemm==0.1.4 \
    peft==0.17.1 \
    diffusers==0.33.0 \
    decord==0.6.0 \
    hyperpyyaml==1.2.2 \
    soundfile==0.12.1 \
    transformers==4.52.4 \
    x_transformers \
    torchdiffeq \
    torchtune \
    torchao==0.13.0 \
    accelerate==1.3.0 \
    onnxruntime \
    jiwer==3.1.0 \
    rich \
    flask \
    flask-cors \
    pypinyin \
    opencc-python-reimplemented \
    loguru

ARG FLASH_ATTN_WHEEL=flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
ARG FLASH_ATTN_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/${FLASH_ATTN_WHEEL}
RUN wget -q ${FLASH_ATTN_URL} -O /tmp/${FLASH_ATTN_WHEEL} && \
    pip install --no-cache-dir /tmp/${FLASH_ATTN_WHEEL} && \
    rm /tmp/${FLASH_ATTN_WHEEL}

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appgroup && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash appuser

WORKDIR /app

RUN npm install -g opencode-ai

# COPY AGENTS-container.md /app/AGENTS.md

RUN opencode --version

CMD ["/bin/bash"]
