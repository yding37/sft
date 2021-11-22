import torch.nn as nn

class FullModelShell(nn.Module):
    def __init__(self, rgb_tokenizer, spec_tokenizer, text_tokenizer, network):
        super().__init__()
        self.rgb_tokenizer = rgb_tokenizer
        self.spec_tokenizer = spec_tokenizer
        self.text_tokenizer = text_tokenizer
        self.network = network

    def forward(self, x_rgb=None, x_spec=None, x_text=None, **kwargs):
        # logging.debug(['input dims', x_rgb.shape, x_spec.shape])

        z_rgb = self.rgb_tokenizer(x_rgb)
        z_spec = self.spec_tokenizer(x_spec)
        z_text = self.text_tokenizer(x_text)

        # logging.debug(['z dims', z_rgb.shape, z_spec.shape])

        logits = self.network(
            z_rgb, z_spec, z_text, **kwargs)

        return logits
