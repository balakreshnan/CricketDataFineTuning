#!/usr/bin/env python3
"""List NGC (nvcr.io) image tags using the enroot credentials, without printing them.
   python3 ngc_tags.py nvidia/ai-dynamo/vllm-runtime [filter]"""
import base64, json, os, re, sys, urllib.request

repo, filt = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
cred = open(os.path.expanduser("~/.config/enroot/.credentials")).read()
m = re.search(r"machine\s+nvcr\.io\s+login\s+(\S+)\s+password\s+(\S+)", cred)
basic = base64.b64encode(f"{m.group(1)}:{m.group(2)}".encode()).decode()
req = urllib.request.Request(f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull",
                             headers={"Authorization": f"Basic {basic}"})
tok = json.load(urllib.request.urlopen(req, timeout=30))["token"]
req = urllib.request.Request(f"https://nvcr.io/v2/{repo}/tags/list?n=2000",
                             headers={"Authorization": f"Bearer {tok}"})
tags = json.load(urllib.request.urlopen(req, timeout=30)).get("tags", [])
tags = [t for t in tags if filt in t and not t.startswith("sha256-")]
print(repo, len(tags), tags[-40:])

# with a third arg "<tag>": print the platforms in that tag's manifest list
if len(sys.argv) > 3:
    req = urllib.request.Request(f"https://nvcr.io/v2/{repo}/manifests/{sys.argv[3]}", headers={
        "Authorization": f"Bearer {tok}",
        "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json"})
    idx = json.load(urllib.request.urlopen(req, timeout=30))
    print(sys.argv[3], [f'{m["platform"]["os"]}/{m["platform"]["architecture"]}' for m in idx.get("manifests", []) if "platform" in m])
