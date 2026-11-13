"""Download integrity, preservation and authentication-boundary checks; no network."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from bird_eval import weights


class WeightDownloadTests(unittest.TestCase):
    payload=b'fixed-test-model-bytes'

    def configuration(self):
        return {'B0':{'checkpoint':'weights/B/B0.pt','sha256':hashlib.sha256(self.payload).hexdigest()}}

    def asset(self):
        return {'B0.pt':{'id':123,'size':len(self.payload),'state':'uploaded','digest':'sha256:'+hashlib.sha256(self.payload).hexdigest()}}

    def test_valid_local_weights_skip_auth_and_network(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(weights,'MODELS',self.configuration()),patch.object(weights,'_token',side_effect=AssertionError('No auth needed')):
            path=Path(folder)/'weights/B/B0.pt';path.parent.mkdir(parents=True);path.write_bytes(self.payload)
            self.assertEqual(weights.fetch_weights(['B0'],Path(folder))[0]['status'],'already_valid')

    def test_unexpected_local_weight_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(weights,'MODELS',self.configuration()),patch.object(weights,'_token',side_effect=AssertionError('No auth needed')):
            path=Path(folder)/'weights/B/B0.pt';path.parent.mkdir(parents=True);path.write_bytes(b'keep me')
            with self.assertRaises(ValueError):weights.fetch_weights(['B0'],Path(folder))
            self.assertEqual(path.read_bytes(),b'keep me')

    def test_valid_download_installs_only_verified_bytes(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(weights,'MODELS',self.configuration()),patch.object(weights,'_token',return_value='test-credential'),patch.object(weights,'_assets',return_value=self.asset()),patch.object(weights,'_open',return_value=io.BytesIO(self.payload)):
            self.assertEqual(weights.fetch_weights(['B0'],Path(folder))[0]['status'],'downloaded_and_verified')
            self.assertEqual((Path(folder)/'weights/B/B0.pt').read_bytes(),self.payload)
            self.assertEqual(list(Path(folder).rglob('*.part')),[])

    def test_bad_download_is_removed_before_install(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(weights,'MODELS',self.configuration()),patch.object(weights,'_token',return_value='test-credential'),patch.object(weights,'_assets',return_value=self.asset()),patch.object(weights,'_open',return_value=io.BytesIO(b'wrong model bytes')):
            with self.assertRaises(ValueError):weights.fetch_weights(['B0'],Path(folder))
            self.assertFalse((Path(folder)/'weights/B/B0.pt').exists())
            self.assertEqual(list(Path(folder).rglob('*.part')),[])

    def test_redirect_does_not_forward_token_to_asset_host(self):
        request=urllib.request.Request('https://api.github.com/repos/x/y/releases/assets/1',headers={'Authorization':'Bearer test-credential'})
        redirect=weights._SafeRedirect().redirect_request(request,None,302,'Found',{},'https://release-assets.githubusercontent.com/asset')
        self.assertIsNone(redirect.get_header('Authorization'))
        with self.assertRaises(ValueError):weights._SafeRedirect().redirect_request(request,None,302,'Found',{},'http://example.com/asset')


if __name__=='__main__':unittest.main()
