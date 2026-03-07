import torch
import logging
from modules import devices
import modules.vae as vae

# =========================
# AMD VAE - ENCODE
# =========================
@torch.no_grad()
def encode(model, x):
    """
    x : [B,3,H,W]  (0-1 range, already scaled outside)
    returns latent : [B,4,H/8,W/8]
    """
    x = x.to(devices.device)
    
    def _encode_fn(a):
        a = a.to(device=devices.device, dtype=devices.dtype_vae)
        
        with torch.autocast("cuda", enabled=not getattr(model, "disable_first_stage_autocast", False)):
            z = model.first_stage_model.encode(a)
        
        if not isinstance(z, torch.Tensor):
            if hasattr(z, "sample"):
                z = z.sample()
        
        return z.float()

    z_tiled = vae.tiled_scale(
        x,
        _encode_fn,
        tile_x=512,
        tile_y=512,
        overlap=64,
        upscale_amount=1/8,
        out_channels=4,
        output_device=devices.device
    )

    if torch.isnan(z_tiled).any():
        logging.warning("AMD VAE Encode produced NaNs. Falling back to Full VAE Encode.")
        z = _encode_fn(x)
    else:
        z = z_tiled

    if hasattr(model, "scale_factor"):
        z = model.scale_factor * z

    return z

# =========================
# AMD VAE - DECODE
# =========================
@torch.no_grad()
def decode(model, z):
    """
    z : [B,4,H,W]
    returns image tensor [-1,1] range
    """
    z = z.to(devices.device)
    
    if hasattr(model, "scale_factor"):
        z = (1.0 / model.scale_factor) * z
        
    def _decode_fn(a):
        a = a.to(device=devices.device, dtype=devices.dtype_vae)
        
        with torch.autocast("cuda", enabled=not getattr(model, "disable_first_stage_autocast", False)):
            out = model.first_stage_model.decode(a)
            
        return out.float()

    image = vae.tiled_scale(
        z,
        _decode_fn,
        tile_x=64,
        tile_y=64,
        overlap=16,
        upscale_amount=8,
        out_channels=3,
        output_device=devices.device
    )

    return image
