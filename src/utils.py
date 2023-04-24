import torch


def center_resize_crop(image, size=224):
    w, h = image.size
    if h < w:
        h, w = size, size * w // h
    else:
        h, w = size * h // w, size

    image = image.resize((w, h))

    box = ((w - size) // 2, (h - size) // 2, (w + size) // 2, (h + size) // 2)
    return image.crop(box)


def encode_image(image, pipe):
    device = pipe._execution_device
    dtype = next(pipe.image_encoder.parameters()).dtype

    if not isinstance(image, torch.Tensor):
        image = pipe.feature_extractor(
            images=image, return_tensors="pt").pixel_values

    image = image.to(device=device, dtype=dtype)
    image_embeds = pipe.image_encoder(image).image_embeds

    return image_embeds

def generate_latents(pipe):
    shape = (1, pipe.unet.in_channels, pipe.unet.config.sample_size,
             pipe.unet.config.sample_size)
    device = pipe._execution_device
    dtype = next(pipe.image_encoder.parameters()).dtype

    return torch.randn(shape, device=device, dtype=dtype)


# https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/4
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1) * \
        low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


def run_remixing(pipe, content_img, style_img, alphas, **kwargs):
    images = []

    content_emb = encode_image(content_img, pipe)
    style_emb = encode_image(style_img, pipe)

    for alpha in alphas:
        emb = slerp(alpha, content_emb, style_emb)
        image = pipe(image=content_img, image_embeds=emb, **kwargs).images[0]
        images.append(image)

    return images