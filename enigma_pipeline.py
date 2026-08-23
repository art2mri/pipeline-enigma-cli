import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime

DEFAULT_INPUT_DIR = "/input_data"
SCT_WORK_DIR = "/home/SCT"
NNUNET_IMAGES_TS = "/home/datav2/nnUNet_raw/Dataset761_SCT/imagesTs"
NNUNET_PREDS = "/home/datav2/inference/761_SCT/preds"

# Configuração das variáveis de ambiente obrigatórias
os.environ["SCT_DIR"] = "/spinalcordtoolbox"
os.environ["PATH"] = "/spinalcordtoolbox/bin:" + os.environ.get("PATH", "")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Enigma-SC pipeline.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory inside the container (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory inside the container.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Process only a specific subject/image. "
            "Can be provided as 'subject01' or 'subject01.nii.gz'. "
            "If omitted, all discovered images will be processed."
        ),
    )
    return parser.parse_args()


def nifti_stem(filename):
    return filename[:-7] if filename.endswith(".nii.gz") else os.path.splitext(filename)[0]


def discover_subjects(input_dir):
    """
    Discover entries without modifying INPUT_DIR.

    Supports:
      1) /input/subject.nii.gz
      2) /input/subject/subject.nii.gz
      3) /input/subject/any_name.nii.gz, as long as there is only one .nii.gz in the folder.

    Returns a list of tuples: (subject, original_image_path).
    """
    subjects = {}

    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)

        if os.path.isfile(item_path) and item.endswith(".nii.gz"):
            subject = nifti_stem(item)
            if subject in subjects:
                raise RuntimeError(f"Duplicate subject detected: {subject}")
            subjects[subject] = item_path
            continue

        if os.path.isdir(item_path):
            expected = os.path.join(item_path, f"{item}.nii.gz")
            if os.path.isfile(expected):
                subject = item
                if subject in subjects:
                    raise RuntimeError(f"Duplicate subject detected: {subject}")
                subjects[subject] = expected
                continue

            candidates = sorted(
                os.path.join(item_path, f)
                for f in os.listdir(item_path)
                if f.endswith(".nii.gz") and os.path.isfile(os.path.join(item_path, f))
            )
            if len(candidates) == 1:
                subject = item
                if subject in subjects:
                    raise RuntimeError(f"Duplicate subject detected: {subject}")
                subjects[subject] = candidates[0]

    return sorted(subjects.items())


def select_subjects(subjects, requested_subject=None):
    """
    Filter the discovered list when --subject is provided.

    Accepts the subject identifier with or without the .nii.gz extension.
    Without requested_subject, returns all subjects.
    """
    if not requested_subject:
        return subjects

    requested = os.path.basename(requested_subject)
    requested_stem = nifti_stem(requested)

    # Primeiro tenta casar com o identificador usado pelo pipeline.
    matches = [(subject, path) for subject, path in subjects if subject == requested_stem]

    # Como conveniência, também aceita o nome real do arquivo NIfTI.
    if not matches:
        matches = [
            (subject, path)
            for subject, path in subjects
            if os.path.basename(path) == requested
            or nifti_stem(os.path.basename(path)) == requested_stem
        ]

    if not matches:
        available = ", ".join(subject for subject, _ in subjects)
        raise ValueError(
            f"Subject/image '{requested_subject}' not found in INPUT_DIR. "
            f"Available: {available or 'none'}"
        )

    if len(matches) > 1:
        matched_names = ", ".join(subject for subject, _ in matches)
        raise ValueError(
            f"The reference '{requested_subject}' is ambiguous. Matches: {matched_names}"
        )

    return matches


def clear_dir(path):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        target = os.path.join(path, name)
        if os.path.isdir(target) and not os.path.islink(target):
            shutil.rmtree(target)
        else:
            os.remove(target)


def copy_working_input(input_file, subject, destination_dir):
    """Copies the original image to an internal workspace with the name expected by the pipeline."""
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, f"{subject}.nii.gz")
    shutil.copy2(input_file, destination)
    return destination


def run_pipeline(input_dir, output_dir, requested_subject=None):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(input_dir):
        print(f"[-] Input directory not found: {input_dir}")
        sys.exit(1)

    # Proteção contra configuração acidental que voltaria a misturar entrada e saída.
    real_input = os.path.realpath(input_dir)
    real_output = os.path.realpath(output_dir)
    if real_input == real_output or os.path.commonpath([real_input, real_output]) == real_input:
        print("[-] OUTPUT_DIR cannot be the same as or inside INPUT_DIR.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[🚀] Starting Enigma-SC Automated Pipeline (Steps 1, 2, 3, 5, 7)")
    print(f"[📥] Input  : {input_dir}")
    print(f"[📤] Output : {output_dir}")

    # ---------------------------------------------------------
    # PASSO 1: Descoberta dos arquivos SEM alterar a entrada
    # ---------------------------------------------------------
    print("\n[1/5] Checking input files...")
    subjects = discover_subjects(input_dir)

    if not subjects:
        print(f"[-] No valid data (.nii.gz) found in {input_dir}. Exiting.")
        sys.exit(0)

    try:
        subjects = select_subjects(subjects, requested_subject)
    except ValueError as exc:
        print(f"[-] {exc}")
        sys.exit(1)

    if requested_subject:
        print(f"[🎯] Individual mode: {subjects[0][0]}")
    else:
        print(f"[📚] Batch mode: all {len(subjects)} image(s) will be processed.")

    # Planilhas globais ficam na raiz do OUTPUT_DIR.
    now = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    csvf = os.path.join(output_dir, f"csa_final_table_{now}.csv")
    eccf = os.path.join(output_dir, f"eccentricity_table_{now}.csv")

    with open(csvf, mode="w", newline="") as f:
        csv.writer(f).writerow(["subject", "CSA C1", "CSA C2", "CSA C3"])
    with open(eccf, mode="w", newline="") as f:
        csv.writer(f).writerow(
            ["subject", "Mean(eccentricity) C1", "Mean(eccentricity) C2", "Mean(eccentricity) C3"]
        )

    # ---------------------------------------------------------
    # LOOP DE PROCESSAMENTO PARA CADA IMAGEM
    # ---------------------------------------------------------
    for subject, input_file in subjects:
        subject_output_dir = os.path.join(output_dir, subject)
        # Cada execução começa com uma pasta limpa para evitar artefatos antigos.
        if os.path.exists(subject_output_dir):
            shutil.rmtree(subject_output_dir)
        os.makedirs(subject_output_dir, exist_ok=True)

        print("\n=========================================")
        print(f"🧠 Processing patient: {subject}")
        print(f"📄 Original: {input_file}")
        print(f"📁 Results: {subject_output_dir}")
        print("=========================================")

        # STEP 2: Spinal Cord Segmentation
        print("[2/5] Spinal Cord Segmentation...")
        clear_dir(SCT_WORK_DIR)
        copy_working_input(input_file, subject, SCT_WORK_DIR)

        subprocess.run("python3 /spine.py", shell=True, check=True, cwd=SCT_WORK_DIR)

        shutil.copy2(
            os.path.join(SCT_WORK_DIR, f"{subject}_seg.nii.gz"),
            subject_output_dir,
        )

        qc_subject = os.path.join(SCT_WORK_DIR, f"qc_{subject}")
        if os.path.exists(qc_subject):
            shutil.copytree(
                qc_subject,
                os.path.join(subject_output_dir, f"qc_{subject}"),
                dirs_exist_ok=True,
            )
        clear_dir(SCT_WORK_DIR)

        # STEP 3: Automatic Vertebral Labeling
        print("[3/5] Automatic Vertebral Labeling...")
        os.makedirs(NNUNET_IMAGES_TS, exist_ok=True)
        os.makedirs(NNUNET_PREDS, exist_ok=True)

        dest_label = os.path.join(NNUNET_IMAGES_TS, f"{subject}_0000.nii.gz")
        pred_file = os.path.join(NNUNET_PREDS, f"{subject}.nii.gz")

        # Evita reutilizar uma predição antiga em caso de falha silenciosa.
        if os.path.exists(pred_file):
            os.remove(pred_file)

        shutil.copy2(input_file, dest_label)

        if subprocess.run("python3 /home/scripts/cuda.py", shell=True).returncode != 0:
            print("      ⚠️ GPU not available for this task. Running on CPU...")
            subprocess.run("python3 /home/scripts/cpu.py", shell=True, check=True)

        shutil.copy2(
            pred_file,
            os.path.join(subject_output_dir, f"{subject}_seg_labeled.nii.gz"),
        )

        if os.path.exists(dest_label):
            os.remove(dest_label)
        if os.path.exists(pred_file):
            os.remove(pred_file)

        # STEP 5: Automatic Spinal Cord Registration
        print("[4/5] Automatic Spinal Cord Registration...")
        clear_dir(SCT_WORK_DIR)
        copy_working_input(input_file, subject, SCT_WORK_DIR)
        shutil.copy2(
            os.path.join(subject_output_dir, f"{subject}_seg.nii.gz"),
            SCT_WORK_DIR,
        )
        shutil.copy2(
            os.path.join(subject_output_dir, f"{subject}_seg_labeled.nii.gz"),
            SCT_WORK_DIR,
        )

        subprocess.run(
            "python3 /spine1.py && python3 /spine2.py && python3 /spine3.py",
            shell=True,
            cwd=SCT_WORK_DIR,
        )

        try:
            shutil.copy2(
                os.path.join(SCT_WORK_DIR, "anat2template.nii.gz"),
                subject_output_dir,
            )
            shutil.copy2(
                os.path.join(SCT_WORK_DIR, f"{subject}_labels_vert.nii.gz"),
                subject_output_dir,
            )
        except FileNotFoundError:
            print("      ⚠️ Warning: Some registration files were not generated correctly.")

        label_dir = os.path.join(SCT_WORK_DIR, "label")
        output_label_dir = os.path.join(subject_output_dir, "label")
        if os.path.exists(label_dir):
            if os.path.exists(output_label_dir):
                shutil.rmtree(output_label_dir)
            shutil.copytree(label_dir, output_label_dir)

        for qc_name in (f"qc_template_{subject}", f"qc_warp_{subject}"):
            qc_source = os.path.join(SCT_WORK_DIR, qc_name)
            if os.path.exists(qc_source):
                shutil.copytree(
                    qc_source,
                    os.path.join(subject_output_dir, qc_name),
                    dirs_exist_ok=True,
                )

        clear_dir(SCT_WORK_DIR)

        # STEP 7: Data Extraction and Calculations
        print("[5/5] Clinical Data Extraction...")
        extraction_dir = os.path.join("/home", subject)
        if os.path.exists(extraction_dir):
            shutil.rmtree(extraction_dir)

        # spine4.py precisa enxergar os resultados anteriores e a imagem anatômica.
        # Criamos uma cópia TEMPORÁRIA em /home; nada é escrito no INPUT_DIR.
        shutil.copytree(subject_output_dir, extraction_dir)
        shutil.copy2(
            input_file,
            os.path.join(extraction_dir, f"{subject}.nii.gz"),
        )

        subprocess.run("python3 /spine4.py", shell=True)

        csa_source = os.path.join(extraction_dir, f"{subject}_csa.csv")
        csa_output = os.path.join(subject_output_dir, f"{subject}_csa.csv")
        if os.path.exists(csa_source):
            shutil.copy2(csa_source, csa_output)

        # Consolidação matemática dos CSVs
        v, k, b1 = [], [subject], [subject]
        try:
            with open(csa_output, "r", encoding="utf-8") as file:
                sp = csv.reader(file)
                for row in sp:
                    v.append(row)
        except FileNotFoundError:
            pass

        for row in v:
            if len(row) > 16 and row[4] in ["1", "2", "3"]:
                try:
                    w = (
                        float(row[6]) * math.cos(math.radians(float(row[8])))
                        + float(row[6]) * math.cos(math.radians(float(row[10])))
                    ) / 2
                    k.append(str(w))
                    b1.append(str(row[16]))
                except (ValueError, IndexError):
                    pass

        with open(csvf, "a", newline="") as f:
            csv.writer(f).writerow(k)
        with open(eccf, "a", newline="") as f:
            csv.writer(f).writerow(b1)

        shutil.rmtree(extraction_dir, ignore_errors=True)

    print(f"\n[✅] Pipeline completed! Results available at: {output_dir}")
    print(f"[🔒] Input directory preserved: {input_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.subject)
