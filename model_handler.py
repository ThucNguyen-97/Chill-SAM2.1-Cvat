import numpy as np
import PIL.Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ModelHandler:
    def __init__(self):
        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "/opt/nuclio/sam2.1_b.pt",
            device="cpu",
        )
        self.predictor = SAM2ImagePredictor(model)

    @staticmethod
    def _mask_to_rle(mask):
        height, width = mask.shape
        pixels = (np.asarray(mask).reshape(-1) != 0).astype(np.uint8)
        changes = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
        rle = np.diff(np.concatenate(([0], changes, [pixels.size]))).tolist()
        if pixels.size and pixels[0] == 1:
            rle.insert(0, 0)
        rle.extend([0, 0, width - 1, height - 1])
        return rle

    @staticmethod
    def _interaction_box(bbox, width, height, pos_points, neg_points):
        if bbox is None:
            points = list(pos_points) + list(neg_points)
            if not points:
                raise ValueError("Provide a bounding box or at least one point")
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        if isinstance(bbox, dict):
            bbox = bbox.get("points", bbox.get("bbox", bbox))
        if len(bbox) == 2 and all(isinstance(point, (list, tuple)) for point in bbox):
            bbox = [bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]]
        values = [float(value) for value in bbox[:4]]
        x1, y1, x2, y2 = values
        return [
            max(0, min(x1, x2)), max(0, min(y1, y2)),
            min(width - 1, max(x1, x2)), min(height - 1, max(y1, y2)),
        ]

    def interact(self, image, bbox, pos_points, neg_points):
        image = image.convert("RGB")
        self.predictor.set_image(np.asarray(image))

        box = None
        if bbox is not None:
            box = np.asarray(self._interaction_box(
                bbox, image.width, image.height, pos_points, neg_points
            ), dtype=np.float32)

        points = list(pos_points) + list(neg_points)
        point_coords = np.asarray(points, dtype=np.float32) if points else None
        point_labels = np.asarray(
            [1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32
        ) if points else None

        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        return self._mask_to_rle(masks[0])

    @staticmethod
    def _encode_state(state):
        return base64.b64encode(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)).decode(
            "ascii"
        )

    @staticmethod
    def _decode_state(state):
        return pickle.loads(base64.b64decode(state))  # nosec B301

    @staticmethod
    def _box(shape):
        if isinstance(shape, dict):
            shape = shape.get("points", shape.get("bbox", shape))
        values = [float(value) for value in shape]
        if len(values) < 4:
            raise ValueError("Tracker shape must contain x1,y1,x2,y2")
        x1, y1, x2, y2 = values[:4]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def infer(self, image, shape, encoded_state):
        prepared = self._preprocess(image)
        if encoded_state is None:
            x1, y1, x2, y2 = self._box(shape)
            mask = np.zeros((prepared["height"], prepared["width"]), dtype=np.float32)
            mask[
                max(0, int(y1)) : min(prepared["height"], int(y2) + 1),
                max(0, int(x1)) : min(prepared["width"], int(x2) + 1),
            ] = 1.0
            resized = torch.nn.functional.interpolate(
                torch.from_numpy(mask)[None, None],
                (self.predictor.image_size, self.predictor.image_size),
                mode="bilinear",
                align_corners=False,
            ).to(self.device)
            predictor_outputs = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": collections.OrderedDict(),
            }
            frame_idx = 0
            current = self._predict(prepared, frame_idx, predictor_outputs, resized)
            predictor_outputs["cond_frame_outputs"][0] = current
        else:
            state = self._decode_state(encoded_state)
            frame_idx = state["frame_idx"] + 1
            predictor_outputs = state["predictor_outputs"]
            current = self._predict(prepared, frame_idx, predictor_outputs)
            predictor_outputs["non_cond_frame_outputs"][frame_idx] = current
            while len(predictor_outputs["non_cond_frame_outputs"]) > self.predictor.num_maskmem:
                predictor_outputs["non_cond_frame_outputs"].popitem(last=False)

        output_mask = (
            torch.nn.functional.interpolate(
                current["pred_masks"],
                size=(prepared["height"], prepared["width"]),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )[0, 0]
            > 0
        ).cpu().numpy().astype(np.uint8)
        ys, xs = np.where(output_mask)
        if len(xs) == 0:
            output_shape = self._box(shape)
        else:
            output_shape = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        state = {"frame_idx": frame_idx, "predictor_outputs": predictor_outputs}
        return output_shape, self._encode_state(state)
