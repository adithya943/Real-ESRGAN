import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np
import time
from statistics import mean

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus'
              'RealESRGANv2-anime-xsx2 | RealESRGANv2-animevideo-xsx2-nousm | RealESRGANv2-animevideo-xsx2'
              'RealESRGANv2-anime-xsx4 | RealESRGANv2-animevideo-xsx4-nousm | RealESRGANv2-animevideo-xsx4'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--half', action='store_true', help='Use half precision during inference')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument('--custom', type=str, default=None, help='Path to custom weights')
    parser.add_argument('--sharpen', action='store_true', help='To sharpen the image prior to SR.')
    parser.add_argument('--prescale', action='store_true', help="Prescale the image before upresing.")
    parser.add_argument('--prescale_val', type=float, default=2.0, help="How much to prescale by.")
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark amount of time the model takes per inference.')
    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in [
            'RealESRGANv2-anime-xsx2', 'RealESRGANv2-animevideo-xsx2-nousm', 'RealESRGANv2-animevideo-xsx2'
    ]:  # x2 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    elif args.model_name in [
            'RealESRGANv2-anime-xsx4', 'RealESRGANv2-animevideo-xsx4-nousm', 'RealESRGANv2-animevideo-xsx4'
    ]:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('realesrgan/weights', args.model_name + '.pth')
    if args.custom:
        model_path = f"experiments/{args.custom}"
    if not os.path.isfile(model_path):
        print(model_path)
        raise ValueError(f'Model {args.model_name} does not exist.')
    time_dict = {
        'setup_time': 0,
        'inference_times': []
    }
    setup_time_start = time.time()
    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half)

    if args.face_enhance:  # Use GFPGAN for face enhancement
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

    # check how long setup takes
    setup_time_end = time.time()
    time_dict['setup_time'] = setup_time_end - setup_time_start

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if args.sharpen:
            img = unsharp_mask(img, amount=2.0)
        if args.prescale:
            img = cv2.resize(img, (0, 0), fx=args.prescale_val, fy=args.prescale_val, interpolation=cv2.INTER_CUBIC)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            inf_time_start = time.time()
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
            inf_time_end = time.time()
            time_dict['inference_times'].append(inf_time_end - inf_time_start)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)

    if args.benchmark:
        print(f"Setup time: {time_dict['setup_time']}")
        print(f"Average inference time: {mean(time_dict['inference_times'])}")


if __name__ == '__main__':
    main()
