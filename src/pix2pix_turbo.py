import sys
import copy
import torch
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig
p = "src/"
sys.path.append(p)
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None
    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1*(1-self.r) + x2*(self.r)


class Pix2Pix_Turbo(torch.nn.Module):
    def __init__(self, name, ckpt_folder="checkpoints"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo",subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder")
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet")

        if name=="sketch_to_image_stochastic":
            # download from url
            p_ckpt = "checkpoint/sketch_to_image_stochastic_lora.pkl"
            sd = torch.load(p_ckpt, map_location="cpu")
            unet_lora_config = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_target_modules"])
            convin_pretrained = copy.deepcopy(unet.conv_in)
            unet.conv_in = TwinConv(convin_pretrained, unet.conv_in)
        
        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        vae_lora_config = LoraConfig(r=sd["rank_vae"], init_lora_weights="gaussian", target_modules=sd["vae_lora_target_modules"])
        vae.decoder.ignore_skip = False
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        unet.add_adapter(unet_lora_config)
        _sd_unet = unet.state_dict()
        for k in sd["state_dict_unet"]: _sd_unet[k] = sd["state_dict_unet"][k]
        unet.load_state_dict(_sd_unet)
        _sd_vae = vae.state_dict()
        for k in sd["state_dict_vae"]: _sd_vae[k] = sd["state_dict_vae"][k]
        vae.load_state_dict(_sd_vae)
        unet.eval()
        vae.eval()
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([999]).long()


    def forward(self, c_t, prompt, deterministic=True, r=1.0, noise_map=None):
        # encode the text prompt
        caption_tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt").input_ids
        caption_enc = self.text_encoder(caption_tokens)[0]
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample()*self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor ).sample).clamp(-1,1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample()*self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control*r + noise_map*(1-r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps, encoder_hidden_states=caption_enc,).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor ).sample).clamp(-1,1)
        return output_image
