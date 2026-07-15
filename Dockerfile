FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy pillow torchvision

COPY model.pth /app/model.pth
COPY prepare_submission.py /app/prepare_submission.py

CMD ["python", "/app/prepare_submission.py"]
