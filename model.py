import torch
from omegaconf import DictConfig, OmegaConf
from models.timm.maxvit_encoder import MaxxVitEncoder
from models.timm.maxvit_decoder import MaxxVitDecoder

yaml_path = './config/MaxVIT.yaml'
cfg = OmegaConf.load(yaml_path)
print(OmegaConf.to_yaml(cfg))

encoder = MaxxVitEncoder(cfg.encoder.maxvit, img_size=cfg.encoder.maxvit.img_size)

encoder_out_chs = encoder.feature_info[-1]['num_chs']
encoder_out_size = cfg.encoder.maxvit.img_size // encoder.feature_info[-1]['reduction']
print("Encode Outputs", encoder_out_chs, encoder_out_size)
print(encoder.feature_info)
decoder = MaxxVitDecoder(cfg.encoder.maxvit, in_chans=encoder_out_chs, input_size=encoder_out_size)
print(decoder.feature_info)
# Output:
sample_tensor = torch.randn(1, 3, 256, 256)
output = encoder(sample_tensor)
print('Encode', output.shape)
output = decoder(output)
print("Decode", output.shape)


