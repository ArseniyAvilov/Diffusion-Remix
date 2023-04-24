from argparse import ArgumentParser
from pathlib import Path

import torch
from PIL import Image

from src.model import StableRemix
from src.utils import run_remixing


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('content_img', type=Path, help='Path to content image')
    parser.add_argument('style_img', type=Path, help='Path to style image')
    parser.add_argument('save_dir', type=Path, nargs='?', default=Path('.'),
                        help='Path to dir where to save remixes')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    if torch.cuda.is_available():
        pipe = StableRemix.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
        )
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    else:
        pipe = StableRemix.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip")
        pipe = pipe.to(device)

    content_img = Image.open(args.content_img).convert('RGB')
    style_img = Image.open(args.style_img).convert('RGB')

    images = run_remixing(pipe, content_img, style_img, [0.65, 0.7, 0.8, 0.9])
    for idx, image in enumerate(images):
        path = args.save_dir / f'remix_{idx}.png'
        print('Saving remix to', path)
        image.save(path)


if __name__ == '__main__':
    main()
