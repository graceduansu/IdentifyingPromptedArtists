from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
    return img


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode(".jpg", img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format="jpeg", quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


def random_jpeg(image, jpeg_prob):
    if np.random.rand() < jpeg_prob:
        quality = np.random.randint(30, 100)
        image = np.array(image)
        if np.random.rand() > 0.5:
            image = cv2_jpg(image, quality)
        else:
            image = pil_jpg(image, quality)
        image = Image.fromarray(image)

    return image


def random_blur(image, blur_prob):
    if np.random.rand() < blur_prob:
        sigma = np.random.uniform(0.0, 3.0)
        image = np.array(image)
        image = gaussian_blur(image, sigma)
        image = Image.fromarray(image)

    return image


def jpeg_at_qual(image, quality):
    image = np.array(image)
    if np.random.rand() > 0.5:
        image = cv2_jpg(image, quality)
    else:
        image = pil_jpg(image, quality)

    image = Image.fromarray(image)
    return image


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """

    if image_prep == "clip_base_randomresizedcrop_hflip_blurplusjpeg0.1":
        # This is the default training transform
        n_px = 224
        img_scale = 0.2
        T = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    n_px,
                    scale=(img_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                _convert_image_to_rgb,
                lambda x: random_blur(x, 0.1),
                lambda x: random_jpeg(x, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif image_prep == "clip_base_noaug":
        n_px = 224
        T = transforms.Compose(
            [
                transforms.Resize(
                    n_px, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(n_px),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    elif image_prep == "dinov2_base_noaug":
        T = transforms.Compose([
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    return T
