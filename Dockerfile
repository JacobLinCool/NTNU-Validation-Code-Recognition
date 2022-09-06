FROM node:slim

RUN apt update && apt install -y python3 python3-pip
RUN python3 -m pip install Pillow numpy tensorflow
RUN npm i -g pnpm

WORKDIR /app
COPY . .
RUN pnpm i

ENTRYPOINT ["node", "index.js"]
