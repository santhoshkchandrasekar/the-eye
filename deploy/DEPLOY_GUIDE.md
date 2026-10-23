# Deploying The EYE: step by step

Free Oracle Cloud ARM server · free DuckDNS address · automatic HTTPS with Caddy. About an afternoon.

## 1 · Build the deployment folder (laptop)
Create a folder `EYE_DEPLOY` holding `m2_eye_app_v35.py`, `eye.service`, a `requirements.txt` and a `project/` folder: a copy of
the project folder the EYE reads, with every detector's `best.pt`, the classifier, the settings and label files and the test photos,
each at its original relative path (Linux is case-sensitive, so spelling and case must match exactly). Write `requirements.txt` from
the EYE's imports without CUDA suffixes such as `+cu121` (the server has no GPU), using `opencv-python-headless`. Start the EYE
once from that folder with `M2_PUBLIC=1` and the GPU hidden, and check that `/api/config` lists **all seven detectors**
(Milestone 2 Runs 1 to 5, boost and boost2) plus `panel_cls: true` for the classifier. Then zip it as `EYE_DEPLOY.zip`. (COMPOUND compares four of them, boost, boost2, Run 5 and
Run 3, plus the classifier; the Model menu offers all seven.)

## 2 · Create the server (Oracle Cloud console)
1. Sign up at cloud.oracle.com (Free Tier; a card is needed for verification). Choose the **home region** carefully:
   it cannot be changed later. A US East region is a good default.
2. Compute → Instances → **Create instance**:
   - Image: **Canonical Ubuntu 24.04** (the aarch64 / ARM build).
   - Shape: **Ampere → VM.Standard.A1.Flex**, **2 OCPUs, 12 GB memory** (the Always Free allowance).
   - Networking: keep the default network; make sure **Assign a public IPv4 address** is on.
   - SSH keys: upload your public key (or let Oracle generate a pair and **download the private key**).
   - Boot volume: 100 GB.
   If it says **out of capacity**, try another availability domain or retry later; this is common and temporary.
3. Open the web ports in Oracle's network: Instance → Subnet → Default Security List → **Add ingress rules**:
   source `0.0.0.0/0`, TCP, destination port **80**; and the same for **443**.
4. Note the instance's **public IP address**.

## 3 · Get a free address (DuckDNS)
1. Sign in at duckdns.org (Google or GitHub).
2. Add a subdomain, for example `the-eye` → `the-eye.duckdns.org`, and set its IP to the server's public IP.
3. Copy your **token** from the top of the page.

## 4 · Install (server)
From your laptop:
```
scp -i <your-key> EYE_DEPLOY.zip setup.sh ubuntu@<public-ip>:~
ssh -i <your-key> ubuntu@<public-ip>
bash setup.sh the-eye <your-duckdns-token>
```
`setup.sh` installs Python and the requirements, opens the server's own firewall (Oracle's Ubuntu blocks everything except
SSH by default), keeps DuckDNS pointed at the server, installs Caddy for HTTPS, and starts the EYE as a service.
It ends by printing the EYE's health and your address.

## 5 · Check
- Open `https://the-eye.duckdns.org` on a laptop and a phone: wake-up, GUIDE, Try this, Beat the EYE, My builders.
- Paste the address into LinkedIn's **Post Inspector** to confirm the preview card.
- Add a free **UptimeRobot** monitor for `https://the-eye.duckdns.org/health`.

## Everyday operations (on the server)
| Task | Command |
|---|---|
| Is it running? | `systemctl status eye` |
| Its log | `journalctl -u eye -n 100 --no-pager` |
| Restart | `sudo systemctl restart eye` |
| Update the EYE | from the laptop: `scp -i <key> m2_eye_app_v35.py ubuntu@<ip>:~/eye/` then `sudo systemctl restart eye` |
| HTTPS problems | `journalctl -u caddy -n 50 --no-pager` (ports 80/443 must be open in both places, step 2.3 and setup.sh) |

## If something goes wrong
- **The page does not load:** check the Oracle security list (step 2.3) and that DuckDNS shows the right IP.
- **Models missing:** `curl 127.0.0.1:7867/api/config` and compare the model list (seven detectors) with the step 1 report.
- **Slow:** expected on CPU. The loader keeps visitors informed. ONNX can help (see DEPLOY_NOTES.md), measured first.
