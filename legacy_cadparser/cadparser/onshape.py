"""Step 1: the only module that talks to Onshape."""

import os
import json
import re
import urllib.parse

import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth


class QuotaExhausted(RuntimeError):
    """Onshape returned HTTP 402: the API quota has run out."""


class OnshapeRequestError(RuntimeError):
    """A request could not reach Onshape."""


class OnshapeClient:
    """Finished for you. Sends authenticated GET requests and counts them."""

    def __init__(self):
        # A stale empty variable in the shell should not mask this project's
        # credentials while running its local tools.
        load_dotenv(override=True)
        self.base_url = os.environ.get("ONSHAPE_API", "https://cad.onshape.com")
        self.auth = HTTPBasicAuth(
            os.environ["ONSHAPE_ACCESS_KEY"], os.environ["ONSHAPE_SECRET_KEY"]
        )
        self.session = requests.Session()
        # Local developer environments commonly retain a proxy from a VPN or
        # debugging tool. Direct HTTPS is the safe default for this local app;
        # organizations that require a proxy can opt back in explicitly.
        self.session.trust_env = os.environ.get("ONSHAPE_USE_ENV_PROXY", "").lower() == "true"
        self.calls = 0

    def get(self, path, params=None):
        self.calls += 1
        try:
            response = self.session.get(
                self.base_url + path,
                auth=self.auth,
                params=params,
                headers={"Accept": "application/json"},
                timeout=60,
            )
        except requests.RequestException as error:
            raise OnshapeRequestError(f"Could not reach Onshape: {error}") from error
        if response.status_code == 402:
            raise QuotaExhausted("Onshape API quota exhausted (HTTP 402).")
        response.raise_for_status()
        return response.json()

    def download_part_mesh(self, ref, link_document_id=None):
        """Download one Onshape part as a meter-scale STL mesh."""
        version_type = "v" if ref.get("documentVersion") else "m"
        version_id = ref.get("documentVersion") or ref.get("documentMicroversion")
        if not version_id:
            raise ValueError("Part reference has no version or microversion.")
        path = (
            f"/api/parts/d/{ref['documentId']}/{version_type}/{version_id}"
            f"/e/{ref['elementId']}/partid/{urllib.parse.quote(ref['partId'], safe='')}/stl"
        )
        params = {"mode": "binary", "units": "meter"}
        if ref.get("configuration"):
            params["configuration"] = ref["configuration"]
        if link_document_id:
            params["linkDocumentId"] = link_document_id
        self.calls += 1
        try:
            response = self.session.get(
                self.base_url + path, auth=self.auth, params=params,
                headers={"Accept": "application/octet-stream"}, timeout=90,
                allow_redirects=False,
            )
            # The export service can redirect from cad.onshape.com to a regional
            # cad-*.onshape.com host. Requests intentionally drops Basic auth on
            # a cross-host redirect, but both hosts are Onshape and the export
            # endpoint still requires the same credentials.
            redirects = 0
            while response.is_redirect:
                location = response.headers.get("Location")
                parsed = urllib.parse.urlparse(location or "")
                if not location or parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".onshape.com"):
                    raise OnshapeRequestError("Onshape returned an unsafe mesh-export redirect.")
                redirects += 1
                if redirects > 5:
                    raise OnshapeRequestError("Too many redirects while downloading a mesh.")
                response = self.session.get(
                    location, auth=self.auth,
                    headers={"Accept": "application/octet-stream"}, timeout=90,
                    allow_redirects=False,
                )
        except requests.RequestException as error:
            raise OnshapeRequestError(f"Could not download part mesh: {error}") from error
        if response.status_code == 402:
            raise QuotaExhausted("Onshape API quota exhausted (HTTP 402).")
        response.raise_for_status()
        if len(response.content) < 84:
            raise ValueError("Onshape returned an empty or invalid STL mesh.")
        return response.content


def parse_url(url):
    """Split an Onshape assembly URL into its IDs.

    https://cad.onshape.com/documents/<did>/w/<wvmid>/e/<eid>
    The middle letter can be w (workspace), v (version), or m (microversion).

    Returns {"did": ..., "wvm": "w", "wvmid": ..., "eid": ...}.
    Raises ValueError if the URL doesn't match or has no element ID.

    Hint: re.search with one capture group per ID.
    """
    match = re.search(
        r"https?://[^/]*onshape\.com/documents/([^/?#]+)/([wvm])/([^/?#]+)/e/([^/?#]+)",
        url,
    )
    if not match:
        raise ValueError(f"Not an Onshape assembly URL: {url!r}")
    did, wvm, wvmid, eid = match.groups()
    return {"did": did, "wvm": wvm, "wvmid": wvmid, "eid": eid}


def fetch_assembly(client, ref, cache_dir, refresh=False):
    """Return the assembly JSON for `ref`, calling the API only if needed.

    1. Build a cache filename from the four IDs in `ref`.
    2. If that file exists and refresh is False, load and return it.
    3. Otherwise call client.get with this path:
         /api/assemblies/d/{did}/{wvm}/{wvmid}/e/{eid}
       and these params:
         {"includeMateFeatures": "true", "includeMateConnectors": "true"}
    4. Create cache_dir if needed, save the JSON, and return it.

    Why the cache matters: every call uses quota. The old project re-fetched
    constantly and saved no raw responses.
    """
    required = ("did", "wvm", "wvmid", "eid")
    if any(key not in ref for key in required):
        raise ValueError("Assembly reference must contain did, wvm, wvmid, and eid.")

    os.makedirs(cache_dir, exist_ok=True)
    cache_name = "{did}_{wvm}_{wvmid}_{eid}.json".format(**ref)
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, encoding="utf-8") as cache_file:
            return json.load(cache_file)

    path = "/api/assemblies/d/{did}/{wvm}/{wvmid}/e/{eid}".format(**ref)
    assembly = client.get(path, params={
        "includeMateFeatures": "true",
        "includeMateConnectors": "true",
    })
    with open(cache_path, "w", encoding="utf-8") as cache_file:
        json.dump(assembly, cache_file, indent=2)
    return assembly
