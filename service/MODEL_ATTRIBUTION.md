# Model attribution and use limits

The unchanged `wav2vec2-small-antideepfake.onnx` asset is attributed to Ge, Wang, et al., NII Yamagishi Laboratory, as recorded in the supplied local model card.

- Original model: [nii-yamagishilab/wav2vec-small-anti-deepfake](https://huggingface.co/nii-yamagishilab/wav2vec-small-anti-deepfake)
- Original code: [NII AntiDeepfake](https://github.com/nii-yamagishilab/AntiDeepfake)
- Paper: [Post-training for Deepfake Speech Detection](https://arxiv.org/abs/2506.21090)
- License stated by the owner and supplied model card: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Model SHA-256: `36ecc56f1f4b1a230e657b14c2d67702b7b01a0d6cb66ac4601497f12c3803d3`

This service does not train, fine-tune, or change the model. It applies the original preprocessing and negative class-1 score convention, then adds development thresholds, validity checks, and a web interface. Attribution must accompany redistribution of the model. Commercial use is excluded. Review the license's complete terms before any redistribution or adaptation; this notice does not grant additional rights to the model or to uploaded audio. No research audio is included in the Docker image.

The supplied local model card is the attribution source; its arena statistics have not been adopted as service evidence. This project's reported error rates come only from the expressly permitted DEV inputs.
