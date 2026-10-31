FROM art2mri/vertebral_labeling:5.0

COPY enigma_pipeline.py /enigma_pipeline.py

ENTRYPOINT ["python3", "/enigma_pipeline.py"]
