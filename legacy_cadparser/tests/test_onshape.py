import pytest

from cadparser.onshape import OnshapeClient, fetch_assembly, parse_url

REF = {"did": "aa11", "wvm": "w", "wvmid": "bb22", "eid": "cc33"}


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        return {"rootAssembly": {"instances": []}}


@pytest.mark.parametrize("letter", ["w", "v", "m"])
def test_parse_url_reads_each_id(letter):
    url = f"https://cad.onshape.com/documents/aa11/{letter}/bb22/e/cc33"
    assert parse_url(url) == {"did": "aa11", "wvm": letter, "wvmid": "bb22", "eid": "cc33"}


def test_parse_url_ignores_extra_text():
    url = "https://cad.onshape.com/documents/aa11/w/bb22/e/cc33?renderMode=0#tab"
    assert parse_url(url)["eid"] == "cc33"


@pytest.mark.parametrize("url", [
    "https://example.com/not-onshape",
    "https://cad.onshape.com/documents/aa11/w/bb22",
])
def test_parse_url_rejects_bad_urls(url):
    with pytest.raises(ValueError):
        parse_url(url)


def test_fetch_requests_the_right_data(tmp_path):
    client = FakeClient()
    fetch_assembly(client, REF, tmp_path)
    path, params = client.calls[0]
    assert path == "/api/assemblies/d/aa11/w/bb22/e/cc33"
    assert params["includeMateFeatures"] == "true"
    assert params["includeMateConnectors"] == "true"


def test_fetch_uses_the_cache_on_the_second_call(tmp_path):
    client = FakeClient()
    first = fetch_assembly(client, REF, tmp_path)
    second = fetch_assembly(client, REF, tmp_path)
    assert first == second
    assert len(client.calls) == 1


def test_refresh_ignores_the_cache(tmp_path):
    client = FakeClient()
    fetch_assembly(client, REF, tmp_path)
    fetch_assembly(client, REF, tmp_path, refresh=True)
    assert len(client.calls) == 2


def test_fetch_creates_the_cache_folder(tmp_path):
    client = FakeClient()
    fetch_assembly(client, REF, tmp_path / "new_folder")
    assert (tmp_path / "new_folder").is_dir()


def test_mesh_export_keeps_auth_when_onshape_redirects_to_a_regional_host():
    class Response:
        def __init__(self, status, location=None, content=b""):
            self.status_code = status
            self.headers = {} if location is None else {"Location": location}
            self.content = content

        @property
        def is_redirect(self):
            return self.status_code in {301, 302, 303, 307, 308}

        def raise_for_status(self):
            assert self.status_code == 200

    class Session:
        def __init__(self):
            self.requests = []

        def get(self, url, **kwargs):
            self.requests.append((url, kwargs))
            if len(self.requests) == 1:
                return Response(302, "https://cad-usw2.onshape.com/export")
            return Response(200, content=b"x" * 84)

    client = OnshapeClient.__new__(OnshapeClient)
    client.base_url, client.auth, client.calls = "https://cad.onshape.com", object(), 0
    client.session = Session()
    result = client.download_part_mesh({
        "documentId": "doc", "documentVersion": "version", "elementId": "element", "partId": "part",
    })
    assert len(result) == 84
    assert len(client.session.requests) == 2
    assert client.session.requests[1][1]["auth"] is client.auth
