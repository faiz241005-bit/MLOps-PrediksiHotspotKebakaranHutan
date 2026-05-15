# MLOps-FireGuard devcontainer base
#
# Base image python:3.11-bullseye punya pre-registered Yarn debian repo
# (/etc/apt/sources.list.d/yarn.list) yang GPG key-nya sudah di-rotate
# (NO_PUBKEY 62D54FD4003F6525). Akibatnya, *setiap* feature devcontainer
# yang trigger `apt-get update` akan gagal — termasuk
# docker-outside-of-docker dan docker-in-docker.
#
# Solusi: hapus repo Yarn yang rusak SEBELUM feature install. Yarn tidak
# kita pakai di project ini, jadi aman untuk dihapus.
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

RUN rm -f /etc/apt/sources.list.d/yarn.list \
          /etc/apt/keyrings/yarnkey.gpg \
          /usr/share/keyrings/yarnkey.gpg 2>/dev/null || true \
 && apt-get update \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
