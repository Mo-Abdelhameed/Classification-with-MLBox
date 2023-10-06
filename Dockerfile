FROM python:3.7.17-slim-bullseye as builder

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dos2unix \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements/requirements.txt /opt/
RUN pip install --upgrade pip
RUN pip install -r /opt/requirements.txt
COPY ./src /opt/src
COPY ./entry_point.sh /opt/
RUN chmod +x /opt/entry_point.sh

WORKDIR /opt/src

RUN chown -R 1000:1000 /opt/src

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/src:${PATH}"
# set non-root user
USER 1000

ENTRYPOINT ["/opt/entry_point.sh"]
