from importlib.metadata import distributions, version, PackageNotFoundError
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
vendor={'torch','torchvision','torchaudio','triton','pytorch-triton'}
errors=[]
for d in distributions():
    for text in d.requires or []:
        r=Requirement(text)
        # Validate base dependencies plus The Well benchmark extras.
        extra='benchmark' if canonicalize_name(d.metadata['Name'])=='the-well' else ''
        if r.marker and not r.marker.evaluate({'extra':extra}): continue
        if canonicalize_name(r.name) in vendor: continue
        try:
            actual=version(r.name)
            if r.specifier and actual not in r.specifier: errors.append(f'{d.metadata["Name"]}: {r} (installed {actual})')
        except PackageNotFoundError: errors.append(f'{d.metadata["Name"]}: missing {r}')
assert not errors, '\n'.join(errors)
import numpy, scipy, h5py, hydra, pandas, matplotlib, plotly, imageio, einops, wandb
print('PASS: offline install and all non-vendor dependency metadata, including The Well benchmark extras')
print('PASS: non-GPU dependency imports; torch/timm/model GPU execution requires cluster validation')
