FROM ghcr.io/prefix-dev/pixi:0.55.0 AS build

# Copy only pixi manifests into the container for environment resolution
COPY pixi.toml pixi.lock /app/
WORKDIR /app

# run the `install` command (or any other). This will also install the dependencies into `/app/.pixi`
RUN pixi install --locked

RUN pixi shell-hook -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
RUN echo 'exec "$@"' >> /app/entrypoint.sh

FROM ubuntu:24.04 AS production
WORKDIR /app
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh
# Copy only the necessary runtime files into the production image
COPY AF2_MiniPAE_JB_v1.py /app/
COPY pixi.toml pixi.lock /app/

# #EXPOSE 8000

# set the entrypoint to the shell-hook script (activate the environment and run the command)
# no more pixi needed in the prod container
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]