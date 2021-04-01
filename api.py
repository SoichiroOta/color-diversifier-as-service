import os
import io

import responder

from color_diversity import (
    load_img, ColorDiversifier, compress_imgs
)


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
IMAGE_FORMAT = env['IMAGE_FORMAT']

api = responder.API(debug=DEBUG)
diversifier = ColorDiversifier()


def diversify(bytes_io):
    img = load_img(bytes_io)
    imgs = diversifier.diversify(img)
    return compress_imgs(imgs, format_=IMAGE_FORMAT)


@api.route("/")
async def diversify_img(req, resp):
    body = await req.content
    resp.content = diversify(
        io.BytesIO(body)
    )


if __name__ == "__main__":
    api.run()