# pipeline-enigma-cli

A Docker-based pipeline for automated processing of NIfTI (`.nii.gz`) spinal cord images.

The container extends the `art2mri/vertebral_labeling:3.0` image and runs `enigma_pipeline.py` as its default entrypoint. The Python orchestrator performs the main Enigma-SC processing steps, including spinal cord segmentation, vertebral labeling, registration, clinical data extraction, and generation of consolidated CSV results.

The input directory is treated as the source of the original NIfTI images, while intermediate and final results are written to a separate output directory.

## Requirements

- [Docker](https://docs.docker.com/get-docker/)
- NVIDIA Container Toolkit and a compatible GPU are optional but recommended for GPU execution.

No local Python environment is required to run the pipeline because the processing environment is provided by the Docker image.

## Installation

Clone this repository:

```bash
git clone https://github.com/art2mri/pipeline-enigma-cli
cd pipeline-enigma-cli
```

Build the Docker image:

```bash
docker build -t pipeline-enigma-cli .
```

## Usage

The container expects:

- an **input** directory containing the original `.nii.gz` images.
- an **output** directory where processing results will be stored.

It is recommended to mount the input directory as read-only (`:ro`) so the original images cannot be modified by the container.

### Process all images

If `--subject` is not provided, all valid NIfTI images found in the input directory are processed.

```bash
docker run --rm \
  --gpus all \
  -v /absolute/path/to/input:/input_data:ro \
  -v /absolute/path/to/output:/output_data \
  pipeline-enigma-cli \
  --input-dir /input_data \
  --output-dir /output_data
```

### Process a single image

Use `--subject` to process only one subject:

```bash
docker run --rm \
  --gpus all \
  -v /absolute/path/to/input:/input_data:ro \
  -v /absolute/path/to/output:/output_data \
  pipeline-enigma-cli \
  --input-dir /dados_entrada \
  --output-dir /output_data \
  --subject patient_001
```

The subject can also be provided with the NIfTI extension:

```bash
--subject patient_001.nii.gz
```

## Input

A simple input directory may look like:

```text
input/
├── patient_001.nii.gz
├── patient_002.nii.gz
└── patient_003.nii.gz
```

The pipeline also supports subject-specific subdirectories.

## Output

Results are organized by subject:

```text
output/
├── patient_001/
│   ├── patient_001_seg.nii.gz
│   ├── patient_001_seg_labeled.nii.gz
│   ├── patient_001_csa.csv
│   └── ...
├── patient_002/
│   └── ...
├── csa_final_table_<timestamp>.csv
└── eccentricity_table_<timestamp>.csv
```

## Command-line options

| Option | Required | Description |
|---|---|---|
| `--input-dir` | No | Input directory inside the container. Default: `/input_data`. |
| `--output-dir` | Yes | Directory where results will be written. |
| `--subject` | No | Process only the specified subject. If omitted, all detected subjects are processed. |

## Contact

For questions, issues, or contributions, please contact:

- **Thiago Junqueira** (*Lab leader*): [thiagojrezende@gmail.com](mailto:thiagojrezende@gmail.com)
- **Diogo Shiraishi**: [diogohshiraishi@gmail.com](mailto:diogohshiraishi@gmail.com)
- **Pedro Marcelino**: [pedronovaesmelo@gmail.com](mailto:pedronovaesmelo@gmail.com)

You can also open an issue in this repository for bug reports or feature requests.
