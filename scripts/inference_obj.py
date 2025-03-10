import tqdm
from latentsync.utils.util import read_video, read_audio, write_video
import torch


class Pipeline:
    def __init__(self):
        pass

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, use_decord=False)
        faces = []
        boxes = []
        affine_matrices = []
        print(f"Affine transforming {len(video_frames)} faces...")
        i = 0
        for frame in tqdm.tqdm(video_frames):
            print("affine transforming...", "%d/%d" % (i, len(video_frames)), video_path)
            i += 1
            face, box, affine_matrix = self.image_processor.affine_transform(frame)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)

        faces = torch.stack(faces)

        return faces, video_frames, boxes, affine_matrices
