import base64
import io
import json

from PIL import Image

from model_handler import ModelHandler


def init_context(context):
    context.logger.info("Initializing custom SAM2 tracker")
    context.user_data.model = ModelHandler()
    context.logger.info("Custom SAM2 tracker initialized")


def handler(context, event):
    data = event.body
    image_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    bbox = data.get("obj_bbox")
    if bbox is None:
        bbox = data.get("bbox")
    pos_points = data.get("pos_points") or []
    neg_points = data.get("neg_points") or []
    rle = context.user_data.model.interact(image, bbox, pos_points, neg_points)

    return context.Response(
        body=json.dumps({
            "shapes": [{
                "points": rle,
                "group": 0,
                "source": "semi-auto",
                "attributes": [],
                "occluded": False,
                "rotation": 0,
                "type": "mask",
            }]
        }),
        headers={},
        content_type="application/json",
        status_code=200,
    )
