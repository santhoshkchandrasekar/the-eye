#!/usr/bin/env bash
# The EYE on an Oracle Cloud Ubuntu 24.04 (ARM) server, with a free DuckDNS address and automatic HTTPS (Caddy).
# Run as the ubuntu user, from the folder holding EYE_DEPLOY.zip:   bash setup.sh <duckdns-subdomain> <duckdns-token>
set -euo pipefail
trap 'echo "!! setup.sh stopped at line $LINENO: $BASH_COMMAND" >&2' ERR    # never fail silently
SUB="${1:?usage: bash setup.sh <duckdns-subdomain> <duckdns-token>}"; TOKEN="${2:?usage: bash setup.sh <duckdns-subdomain> <duckdns-token>}"
DOMAIN="$SUB.duckdns.org"; EYE=/home/ubuntu/eye; export DEBIAN_FRONTEND=noninteractive
echo "== 1/6 system packages"; sudo apt-get update -q && sudo apt-get install -yq python3-venv python3-pip unzip curl gnupg iptables-persistent netfilter-persistent libgl1 libglib2.0-0
echo "== 2/6 the EYE's files";  mkdir -p "$EYE"; unzip -oq EYE_DEPLOY.zip -d "$EYE"
echo "== 3/6 Python (CPU)";     python3 -m venv "$EYE/.venv"; "$EYE/.venv/bin/pip" install -q --upgrade pip; "$EYE/.venv/bin/pip" install -q -r "$EYE/requirements.txt"
# Ultralytics pulls in the desktop OpenCV, which needs a display library a server lacks: keep only the headless build
HEADLESS=$(grep -i '^opencv-python-headless' "$EYE/requirements.txt" || echo opencv-python-headless)
"$EYE/.venv/bin/pip" uninstall -y -q opencv-python 2>/dev/null || true
"$EYE/.venv/bin/pip" install -q --force-reinstall --no-deps "$HEADLESS"
"$EYE/.venv/bin/python" -c "import cv2, torch, ultralytics; print('   ok: OpenCV', cv2.__version__, '· torch', torch.__version__, '· ultralytics', ultralytics.__version__)"
echo "== 4/6 firewall";         # Oracle's Ubuntu images block everything except SSH; open web ports 80 and 443
for P in 80 443; do   # place each rule just above Oracle's catch-all REJECT, wherever it is (and move it there if an earlier run put it below)
  while sudo iptables -D INPUT -p tcp --dport "$P" -m state --state NEW -j ACCEPT 2>/dev/null; do :; done
  N=$(sudo iptables -L INPUT --line-numbers -n | awk '$2 == "REJECT" { print $1; exit }')
  sudo iptables -I INPUT "${N:-1}" -p tcp --dport "$P" -m state --state NEW -j ACCEPT
done
sudo iptables -L INPUT --line-numbers -n | sed 's/^/   /'
sudo netfilter-persistent save
echo "== 5/6 DuckDNS + HTTPS";  mkdir -p ~/duckdns
printf '#!/bin/sh\ncurl -s "https://www.duckdns.org/update?domains=%s&token=%s&ip=" -o ~/duckdns/duck.log\n' "$SUB" "$TOKEN" > ~/duckdns/duck.sh; chmod 700 ~/duckdns/duck.sh; ~/duckdns/duck.sh
echo "   DuckDNS says: $(cat ~/duckdns/duck.log)"
{ crontab -l 2>/dev/null | grep -v duck.sh || true; echo "*/5 * * * * ~/duckdns/duck.sh >/dev/null 2>&1"; } | crontab -   # a fresh server has no crontab yet
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt-get update -q && sudo apt-get install -yq caddy
printf '%s {\n\tencode gzip\n\treverse_proxy 127.0.0.1:7867\n}\n' "$DOMAIN" | sudo tee /etc/caddy/Caddyfile >/dev/null; sudo systemctl restart caddy
echo "== 6/6 the EYE service";  sudo cp "$EYE/eye.service" /etc/systemd/system/eye.service; sudo systemctl daemon-reload; sudo systemctl enable --now eye
echo "   waiting for the EYE to load its models (a minute or two on CPU)..."
for i in $(seq 1 72); do if curl -sf 127.0.0.1:7867/health >/dev/null; then curl -s 127.0.0.1:7867/health; echo; break; fi; sleep 5; done
echo; echo "Done. Open https://$DOMAIN  (the HTTPS certificate can take a minute on first visit)."
