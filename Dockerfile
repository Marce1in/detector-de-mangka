FROM node:22-bookworm

WORKDIR /app

ENV PNPM_HOME="/pnpm"
ENV PATH="${PNPM_HOME}:${PATH}"
ENV UV_LINK_MODE=copy

RUN corepack enable \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        curl \
        ca-certificates \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && rm -rf /var/lib/apt/lists/*

COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

EXPOSE 3000

CMD ["pnpm", "start"]
