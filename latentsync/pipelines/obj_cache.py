image_processor = None
from ..utils.image_processor import ImageProcessor


def get_image_processor(height, mask):
    global image_processor
    if image_processor is None:
        image_processor = ImageProcessor(height, mask=mask, device="cuda")
    return image_processor


