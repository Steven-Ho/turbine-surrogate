import pathlib, subprocess, zipfile, urllib.request, tarfile, hashlib, json
out=pathlib.Path('/out'); wheels=out/'wheelhouse'; wheels.mkdir(exist_ok=True)
work=pathlib.Path('/tmp/pde-fm-build'); work.mkdir(exist_ok=True)
def run(*args): subprocess.run(args,check=True)
# Metadata-only placeholders constrain the resolver; these are NEVER shipped.
stubs=work/'vendor-stubs'; stubs.mkdir(exist_ok=True)
for name,version in [('torch','2.6.0'),('torchvision','0.21.0'),('torchaudio','2.6.0'),('triton','3.2.0')]:
    version += '+vendorplaceholder'
    info=f'{name}-{version}.dist-info'
    with zipfile.ZipFile(stubs/f'{name}-{version}-py3-none-any.whl','w') as z:
        z.writestr(info+'/METADATA',f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
        z.writestr(info+'/WHEEL','Wheel-Version: 1.0\nGenerator: pde-fm\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        z.writestr(info+'/RECORD','')
commit='ad50de0e8861380d7c4d62c0503c654710f026e7'
archive=work/'the_well.tar.gz'
print('Downloading pinned The Well source',flush=True)
with urllib.request.urlopen(f'https://codeload.github.com/PolymathicAI/the_well/tar.gz/{commit}',timeout=120) as response:
    archive.write_bytes(response.read())
with tarfile.open(archive) as tf:
    for member in tf.getmembers():
        assert (work/member.name).resolve().is_relative_to(work.resolve())
    tf.extractall(work, members=[m for m in tf.getmembers() if m.isfile() or m.isdir()])
source=next(work.glob('the_well-*'))
run('python','-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),str(source))
req=pathlib.Path('/source/requirements-metax.txt').read_text()
req='\n'.join(line for line in req.splitlines() if not line.startswith('the_well['))+'\nthe_well[benchmark]\nsetuptools>=68\nwheel\n'
(out/'requirements-pde-fm-offline.txt').write_text(req)
constraints=work/'constraints.txt'; constraints.write_text('torch==2.6.0\ntorchvision==0.21.0\ntorchaudio==2.6.0\ntriton==3.2.0\n')
run('python','-m','pip','wheel','--no-deps','--wheel-dir',str(wheels),'antlr4-python3-runtime==4.9.3')
run('python','-m','pip','download','--only-binary=:all:','--find-links',str(stubs),'--find-links',str(wheels),'-c',str(constraints),'-r',str(out/'requirements-pde-fm-offline.txt'),'-d',str(wheels))
for p in wheels.glob('*.whl'):
    if p.name.split('-')[0].lower() in {'torch','torchvision','torchaudio','triton'} or p.name.startswith('nvidia_'): p.unlink()
(out/'BUILD_INFO.json').write_text(json.dumps({'target':'CPython 3.10 Linux x86_64','the_well_commit':commit,'the_well_source_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'vendor_packages_excluded':['torch','torchvision','torchaudio','triton','nvidia-*'],'note':'Resolver placeholders are not vendor binaries. Validate against actual cluster packages.'},indent=2)+'\n')
print('WHEELHOUSE COMPLETE',flush=True)
