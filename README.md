# Chill-SAM2.1-Cvat

Custom SAM2.1 Base+ interactor for CVAT Community/Nuclio.

This function supports interactive segmentation from a bounding box, positive
points, or negative points. It returns a CVAT mask response for use with the
CVAT AI Tools interactor.

## Contents

- `main.py` — Nuclio HTTP handler.
- `model_handler.py` — SAM2.1 inference and CVAT interaction handling.
- `function.yaml` — Nuclio deployment definition.
- `function-runtime.yaml` — runtime configuration.
- `Dockerfile` — container build metadata.
- `sam2.1_b.pt` — SAM2.1 Base+ checkpoint; tracked with Git LFS.

## Requirements

- Docker Desktop
- CVAT Community running locally
- Nuclio/serverless enabled in CVAT
- Git LFS for cloning the checkpoint

## Deploy

1. Clone this repository with Git LFS enabled.
2. Start CVAT and its serverless components.
3. Deploy `function.yaml` using the CVAT/Nuclio deployment workflow.
4. In a CVAT job, open **AI Tools → Interactors** and select
   `SAM2.1 Base+ Custom Interactor`.

The function is CPU-compatible but may be slow. A GPU-enabled deployment can
be added later if the team needs faster interactive annotation.

## Annotation note

SAM proposes a mask; the annotator must still review the visible boundary,
remove unwanted regions, and follow the competition's class-specific rules.
SAM output is not a replacement for human review.

## License and model terms

The custom handler/configuration and the SAM2.1 checkpoint may have different
licenses. Check the upstream SAM2 license and checkpoint terms before public
distribution. Do not commit passwords, API keys, private images, or CVAT
database files.
