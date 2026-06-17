# Fedora-native base (mandate: Fedora base OS for every part of the solution)
FROM registry.fedoraproject.org/fedora:42

WORKDIR /app

# python3 + build deps via dnf (git for git+ installs; gcc/python3-devel for any sdist builds).
# python3-lxml: Fedora-native lxml so yfinance get_earnings_dates() HTML scrape works
# out of the box (no per-run `pip install lxml` — see foreman ANSWER-014/015).
RUN dnf -y install python3 python3-pip python3-devel gcc git python3-lxml \
    && dnf clean all && rm -rf /var/cache/dnf
RUN pip install --upgrade pip

COPY . .
RUN pip install --no-cache-dir --timeout 120 ".[full]"
