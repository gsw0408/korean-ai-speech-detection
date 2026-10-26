#!/bin/bash
# Instance setup: download inputs via presigned URLs, verify hashes, build two Python environments.
set -euo pipefail
mkdir -p /opt/exp && cd /opt/exp
while read -r name url; do
  curl -sSfL --retry 5 -o "$name" "$url"
done < /opt/exp/urls.txt
sha256sum -c inputs.sha256
mkdir -p code assets
unzip -q -o code.zip -d code
unzip -q -o ft_data.zip -d .
mv -f WavLM-Large.pt bam_checkpoint.zip nii_frames_dyn.onnx assets/
apt-get -qq update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3.10-venv libsndfile1 >/dev/null
python3 -m venv /opt/bam
/opt/bam/bin/pip -q install --upgrade "pip<24.1"
/opt/bam/bin/pip -q install --no-deps -r code/bam_requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
python3 -m venv /opt/nii
/opt/nii/bin/pip -q install onnxruntime-gpu==1.19.2 numpy==1.26.4 soundfile==0.12.1 nvidia-cudnn-cu12==9.1.0.70 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
/opt/bam/bin/python -c "import torch,lightning,s3prl;print('bam env', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_OK
