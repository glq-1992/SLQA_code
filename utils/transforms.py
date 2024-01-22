from __future__ import division
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import Compose, Lambda
import numpy as np
import numbers
import torch

class ToTensor(object):

    def __call__(self, images):
        tensor = []
        for image in images:
            tensor.append(F.to_tensor(image))
        return tensor
        # return torch.stack(tensor, dim=0)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    def __call__(self, tensor):
        batch_size = len(tensor)
        res = []
        for i in range(0, batch_size):
            res.append(F.normalize(tensor[i], self.mean, self.std, self.inplace))
        # return torch.stack(res, dim=0)
        return res

class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        result = []
        for img in imgs:
            img = Image.fromarray(img)
            grayed = F.to_grayscale(img, num_output_channels=self.num_output_channels)
            result.append(np.asarray(grayed))

        return result

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        # assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, iamges):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        res = []
        for image in iamges:
            res.append(F.resize(image, self.size, self.interpolation))
        return res


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        res = []
        for image in imgs:
            if self.padding is not None:
                image = F.pad(image, self.padding, self.fill, self.padding_mode)

            # pad the width if needed
            if self.pad_if_needed and image.size[0] < self.size[1]:
                image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and image.size[1] < self.size[0]:
                image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

            i, j, h, w = self.get_params(image, self.size)
            res.append(F.crop(image, i, j, h, w))
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, images):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        res = []
        flag = torch.rand(1) < self.p
        for image in images:
            if flag:
                res.append(F.hflip(image))
            else:
                res.append(image)
        return res


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class MyRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, outsize):
        depth, h, w = img.shape
        th, tw = outsize

        if w == tw and h == th:
            return 0, 0, th, tw

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return i, j, th, tw

    def __call__(self, imgs):
        """
        :param imgs: seq_len, image_channel, image_height, image_width
        :return:
        """
        cropped_imgs = []
        for img in imgs:
            i, j, h, w = self.get_params(img, self.size)
            cropped_imgs.append(img[:, i:i + h, j:j + w])

        return cropped_imgs


class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        depth, h, w = img.shape
        th, tw = output_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            imgs: seq_len, image_channel, image_height, image_width

        Returns:
            Cropped image.
        """
        cropped_imgs = []
        for img in imgs:
            i, j, h, w = self.get_params(img, self.size)
            cropped_imgs.append(img[:, i:i + h, j:j + w])
        return cropped_imgs


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, including_size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.including_size = including_size

    @staticmethod
    def get_params(img, output_size, including_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        depth, h, w = img.shape
        th, tw = output_size
        ih, iw = including_size
        if w == tw and h == th:
            return 0, 0, h, w
        top_smallest = int(h / 2 + ih / 2 - th)
        top_largest = int(h / 2 - ih / 2)

        left_smallest = int(w / 2 + iw / 2 - tw)
        left_largest = int(w / 2 - iw / 2)

        top_smallest = max(top_smallest, 0)
        top_largest = min(top_largest, h - th)
        left_smallest = max(left_smallest, 0)
        left_largest = min(left_largest, w - tw)
        i = random.randint(top_smallest, top_largest)
        j = random.randint(left_smallest, left_largest)
        return i, j, th, tw

    def __call__(self, imgs):
        """
        Args:
            imgs list: seq_len, image_channel, image_height, image_width.

        Returns:
            list of PIL Image: Cropped images.
        """
        cropped_imgs = []
        for img in imgs:
            i, j, h, w = self.get_params(img, self.size, self.including_size)
            cropped_imgs.append(img[:, i:i + h, j:j + w])

        return cropped_imgs


# class ColorJitter(object):
#     """Randomly change the brightness, contrast and saturation of an image.
#
#     Args:
#         brightness (float): How much to jitter brightness. brightness_factor
#             is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
#         contrast (float): How much to jitter contrast. contrast_factor
#             is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
#         saturation (float): How much to jitter saturation. saturation_factor
#             is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
#         hue(float): How much to jitter hue. hue_factor is chosen uniformly from
#             [-hue, hue]. Should be >=0 and <= 0.5.
#     """
#
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation
#         self.hue = hue
#
#     @staticmethod
#     def get_params(brightness, contrast, saturation, hue):
#         """Get a randomized transform to be applied on image.
#
#         Arguments are same as that of __init__.
#
#         Returns:
#             Transform which randomly adjusts brightness, contrast and
#             saturation in a random order.
#         """
#         transforms = []
#         if brightness > 0:
#             # brightness_factor = np.random.uniform(1, 1 + brightness)
#             brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
#             transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
#
#         if contrast > 0:
#             # contrast_factor = np.random.uniform(1, 1 + contrast)
#             contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
#             transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
#
#         if saturation > 0:
#             saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
#             transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
#
#         if hue > 0:
#             hue_factor = np.random.uniform(-hue, hue)
#             transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))
#
#         np.random.shuffle(transforms)
#         transform = Compose(transforms)
#
#         return transform
#
#     def __call__(self, imgs):
#         """
#         Args:
#             img (PIL Image): Input image.word
#
#         Returns:
#             PIL Image: Color jittered image.
#         """
#         transform = self.get_params(self.brightness, self.contrast,
#                                     self.saturation, self.hue)
#         result = []
#         for img in imgs:
#             img = Image.fromarray(img)
#             transformed = transform(img)
#             result.append(np.asarray(transformed))
#
#         return result
#
#
#
#
# class Crop(object):
#     """Crop the given PIL Image at a given location.
#
#         Args:
#             size (sequence or int): Desired output size of the crop. If size is an
#                 int instead of sequence like (h, w), a square crop (size, size) is
#                 made.
#             left (int): left bounding box
#             top (int): top bounding box
#         """
#
#     def __init__(self, size, left, top):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.left = left
#         self.top = top
#
#     def __call__(self, imgs):
#         """
#         Args:
#             imgs (list of PIL Image): Images to be cropped.
#
#         Returns:
#             list of PIL Image: Cropped images.
#         """
#         h, w = self.size
#
#         return [img[self.top:self.top + h, self.left:self.left + w, :] for img in imgs]
#
#
#
#
# class RandomHorizontalFlip(object):
#     """Horizontally flip the given PIL Image randomly with a probability of 0.5."""
#     """corrected"""
#
#     def __call__(self, imgs):
#         """
#         Args:
#             imgs list: seq_len, image_channel, image_height, image_width
#
#         Returns:
#             np.array: Randomly flipped image.
#         """
#
#         fliped_imgs = []
#         if random.random() < 0.5:
#             fliped = True
#         else:
#             fliped = False
#
#         for img in imgs:
#             if fliped:
#                 fliped_imgs.append(img[:, :, ::-1])
#             else:
#                 fliped_imgs.append(img)
#         return imgs
