"""Download fixed private-release weights, verify SHA-256, then install atomically."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from .settings import MODELS,ROOT

REPOSITORY='BillWang2002/COMP9517_Evaluation'
RELEASE_TAG='weights-v1.0'
API=f'https://api.github.com/repos/{REPOSITORY}'


def _sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _token():
    # Use an existing login. Never save credentials or put them in download URLs.
    value=os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if value:return value.strip()
    try:
        result=subprocess.run(['git','credential','fill'],input=f'url=https://github.com/{REPOSITORY}.git\n\n',
                              cwd=ROOT,text=True,capture_output=True,timeout=30,
                              env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
    except (OSError,subprocess.TimeoutExpired):result=None
    if result is not None and result.returncode==0:
        fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        if fields.get('password'):return fields['password']
    raise ValueError('Private weights require GitHub repository access. Configure Git HTTPS login or set GITHUB_TOKEN; never commit the token.')


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,fp,code,msg,headers,newurl):
        if urllib.parse.urlsplit(newurl).scheme!='https':raise ValueError('Refusing a non-HTTPS weight redirect.')
        redirected=super().redirect_request(request,fp,code,msg,headers,newurl)
        if redirected is not None and urllib.parse.urlsplit(request.full_url).netloc.lower()!=urllib.parse.urlsplit(newurl).netloc.lower():
            redirected.remove_header('Authorization')
        return redirected


def _open(url,token,accept):
    request=urllib.request.Request(url,headers={'Authorization':f'Bearer {token}','Accept':accept,
        'X-GitHub-Api-Version':'2026-03-10','User-Agent':'COMP9517-Evaluation-Weights'})
    return urllib.request.build_opener(_SafeRedirect()).open(request,timeout=120)


def _assets(token):
    with _open(f'{API}/releases/tags/{RELEASE_TAG}',token,'application/vnd.github+json') as response:release=json.load(response)
    if release.get('tag_name')!=RELEASE_TAG or release.get('draft'):
        raise ValueError('The fixed weights release is not published.')
    return {a['name']:a for a in release['assets'] if a.get('state')=='uploaded'}


def _install(asset,path,expected,token):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.part',delete=False) as stream:
            temporary=Path(stream.name);digest=hashlib.sha256();count=0
            with _open(f"{API}/releases/assets/{int(asset['id'])}",token,'application/octet-stream') as response:
                for block in iter(lambda:response.read(1024*1024),b''):
                    stream.write(block);digest.update(block);count+=len(block)
        if count!=int(asset['size']) or digest.hexdigest()!=expected:
            raise ValueError(f'Weight size/SHA-256 mismatch: {path.name}; partial download removed.')
        if path.exists():
            if _sha256(path)!=expected:raise ValueError(f'Preserving an unexpected existing weight: {path}')
        else:temporary.replace(path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def fetch_weights(models=('A1','B0'),root=ROOT):
    models=list(models)
    if not models or len(set(models))!=len(models) or any(m not in MODELS for m in models):raise ValueError('Choose unique A1/B0 model names.')
    paths={m:Path(root)/MODELS[m]['checkpoint'] for m in models};missing=[];rows=[]
    for model,path in paths.items():
        if path.exists():
            if _sha256(path)!=MODELS[model]['sha256']:raise ValueError(f'Existing weight has a different SHA-256; move it before retrying: {path}')
        else:missing.append(model)
    try:
        if missing:
            token=_token();assets=_assets(token)
            for model in missing:
                filename=paths[model].name
                if filename not in assets:raise ValueError(f'Missing published asset: {filename}')
                digest=assets[filename].get('digest')
                if digest and digest!='sha256:'+MODELS[model]['sha256']:raise ValueError(f'Release checksum differs for {model}.')
            for model in missing:
                print(f'Downloading {model} from private release {RELEASE_TAG}...',flush=True)
                _install(assets[paths[model].name],paths[model],MODELS[model]['sha256'],token)
    except urllib.error.HTTPError as exc:
        raise ValueError(f'GitHub HTTP {exc.code}: verify private repository access and that {RELEASE_TAG} is published.') from None
    except urllib.error.URLError as exc:
        raise ValueError(f'GitHub weight download failed: {exc.reason}') from None
    for model,path in paths.items():
        rows.append({'model':model,'path':str(path.resolve()),'sha256':_sha256(path),
                     'status':'downloaded_and_verified' if model in missing else 'already_valid'})
    return rows
