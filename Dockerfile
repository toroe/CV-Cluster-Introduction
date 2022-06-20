FROM python:latest

ENV TERM xterm

RUN apt-get update
RUN apt-get update && DEBIAN_FRONTEND="noninteractive"\
    apt-get install -y \
    rsync htop tmux git \
    libopenmpi-dev openssh-server ssh \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -yq install openssh-server; \
  mkdir -p /var/run/sshd; \
  mkdir /root/.ssh && chmod 700 /root/.ssh; \
  touch /root/.ssh/authorized_keys

RUN pip3 install --upgrade pip matplotlib numpy opencv-contrib-python scikit-learn scipy torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113





COPY sshd_config /etc/ssh/sshd_config

EXPOSE 22
COPY src .
#ENTRYPOINT ["ssh-start"]
CMD ["/usr/sbin/sshd", "-D"]



