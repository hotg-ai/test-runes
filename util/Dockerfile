FROM rust:latest as builder

WORKDIR /app

RUN apt-get update -y

RUN apt-get install parallel

RUN apt-get install unzip

RUN curl -L https://github.com/hotg-ai/rune/releases/download/nightly/rune.x86_64-unknown-linux-gnu.zip \
    -o rune.x86_64-unknown-linux-gnu.zip

RUN ls

RUN unzip rune.x86_64-unknown-linux-gnu.zip

RUN chmod a+x rune

FROM ubuntu:latest

WORKDIR /app

COPY --from=builder /app/rune .



#CMD ["sh"]
