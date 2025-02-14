from latentsync.utils.util import read_video,save_image
from PIL import Image

frames = read_video("tmp.mp4", change_fps=False)
from preprocess.filter_high_resolution import FaceDetector
fd = FaceDetector()
i = 0
for frame in frames:
    if not fd.detect_face(frame):
        image = Image.fromarray(frame)

        # 保存为图片文件
        image.save("output_image.png")  # 支持格式：PNG, JPEG, BMP 等
        print(i)
    i += 1