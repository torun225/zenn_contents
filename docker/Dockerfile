FROM node:14

ARG USER_NAME
ARG GROUP_NAME
ARG USER_UID
ARG USER_GID

RUN deluser --remove-home node \
  && addgroup --gid $USER_GID $GROUP_NAME \
  && adduser --uid $USER_UID --shell /bin/sh $USER_NAME --ingroup $GROUP_NAME

RUN npm init --yes \
  && npm install -g zenn-cli@latest \
  && npx zenn init

RUN mkdir /home/$USER_NAME/contents \
  && chown -R ${GROUP_NAME}:${USER_NAME} /home/$USER_NAME/contents

USER $USER_NAME
WORKDIR /home/$USER_NAME/contents