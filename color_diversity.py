import io
from zipfile import ZipFile, ZIP_DEFLATED
import itertools

import numpy as np
from PIL import Image as pil_image


def load_img(bytes_io):
    return pil_image.open(bytes_io)


def save_img(img_array, format_='PNG', mode='RGB'):
    image = pil_image.fromarray(
        np.uint8(img_array),
        mode
    ).convert('RGB')
    with io.BytesIO() as bytes_io:
        image.save(bytes_io, format=format_)
        return bytes_io.getvalue()


def compress_imgs(imgs, format_='PNG', mode='RGB'):
    zip_stream = io.BytesIO()

    # ファイルに書き出す代わりに zip_stream に zip 圧縮したデータを出力
    with ZipFile(zip_stream, 'w', compression=ZIP_DEFLATED) as new_zip:
        for i, img in enumerate(imgs):
            img_bytes = save_img(img, format_, mode=mode)
            new_zip.writestr(f"{i}.{format_.lower()}", img_bytes)

    # 生成された zip 圧縮データを出力
    return zip_stream.getvalue()


class ColorDiversifier:
    def _inverse_rgb(self, img_array, rgb_flags=None):
        complemented_rgb_flags = [
            True, True, True
        ] if rgb_flags is None else rgb_flags
        inversed_img_array = img_array.copy()
        for i, flag in enumerate(complemented_rgb_flags):
            if flag:
                inversed_img_array[:, :, i] = 255.0 - inversed_img_array[:, :, i]
        return inversed_img_array

    def _inverse_cmyk(self, img_array, cmyk_flags=None):
        complemented_cmyk_flags = [
            True, True, True, True
        ] if cmyk_flags is None else cmyk_flags
        inversed_img_array = img_array.copy()
        for i, flag in enumerate(complemented_cmyk_flags):
            if flag:
                inversed_img_array[:, :, i] = 255.0 - inversed_img_array[:, :, i]
        return inversed_img_array

    def _swap_rgb(self, img_array, rgb_order=None):
        if rgb_order is None:
            return img_array
        rgb_dict = dict(
            r=img_array[:, :, 0].copy(),
            g=img_array[:, :, 1].copy(),
            b=img_array[:, :, 2].copy()
        )
        swapped_img_array = img_array.copy()
        for i, c in enumerate(rgb_order):
            swapped_img_array[:, :, i] = rgb_dict[c]
        return swapped_img_array

    def _swap_cmyk(self, img_array, cmyk_order=None):
        if cmyk_order is None:
            return img_array
        cmyk_dict = dict(
            c=img_array[:, :, 0].copy(),
            m=img_array[:, :, 1].copy(),
            y=img_array[:, :, 2].copy()
        )
        swapped_img_array = img_array.copy()
        for i, c in enumerate(cmyk_order):
            swapped_img_array[:, :, i] = cmyk_dict[c]
        return swapped_img_array

    def _diversify_with_rgb(self, img_array):
        tf_list = [False, True]
        rgb_flags_list = [[r, g, b] for r in tf_list for g in tf_list for b in tf_list]
        imgs = [
            self._swap_rgb(
                self._inverse_rgb(
                    img_array, rgb_flags
                ),
                rgb_order
            ) for rgb_flags in rgb_flags_list for rgb_order in itertools.permutations(
                ['r', 'g', 'b'], 3
            ) 
        ]
        return imgs

    def _diversify_with_cmyk(self, img_array):
        tf_list = [False, True]
        cmyk_flags_list = [[c, m, y, k] for c in tf_list for m in tf_list for y in tf_list for k in tf_list]
        imgs = [
            self._swap_cmyk(
                self._inverse_cmyk(
                    img_array, cmyk_flags
                ),
                cmyk_order
            ) for cmyk_flags in cmyk_flags_list for cmyk_order in itertools.permutations(
                ['c', 'm', 'y'], 3
            ) 
        ]
        return imgs
        
    def diversify(self, img, mode='RGB'):
        raw_img_array = np.array(img.convert(mode), dtype=np.float64)
        if mode == 'CMYK':
            imgs = self._diversify_with_cmyk(raw_img_array)
        else:
            imgs = self._diversify_with_rgb(raw_img_array)
            gray_scale_img_array = np.array(img.convert('L').convert('RGB'), dtype=np.float64)
            imgs.extend(
                [
                    gray_scale_img_array,
                    self._inverse_rgb(gray_scale_img_array)
                ]
            )
        return imgs


