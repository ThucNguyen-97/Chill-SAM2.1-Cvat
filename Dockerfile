FROM cvat.pth.facebookresearch.sam2.interactor:debug

COPY sam2.1_b.pt /opt/nuclio/sam2.1_b.pt
COPY model_handler.py /opt/nuclio/model_handler.py
COPY main.py /opt/nuclio/main.py
