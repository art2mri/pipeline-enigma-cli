import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime

DEFAULT_INPUT_DIR = "/dados_entrada"
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
    parser = argparse.ArgumentParser(
        description="Pipeline automatizada Enigma-SC com entrada somente leitura e saída separada."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Diretório de entrada dentro do contêiner (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Diretório de saída dentro do contêiner.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Processa somente um paciente/imagem específica. "
            "Pode ser informado como 'paciente01' ou 'paciente01.nii.gz'. "
            "Se omitido, todas as imagens encontradas serão processadas."
        ),
    )
    return parser.parse_args()


def nifti_stem(filename):
    return filename[:-7] if filename.endswith(".nii.gz") else os.path.splitext(filename)[0]


def discover_subjects(input_dir):
    """
    Descobre entradas sem modificar INPUT_DIR.

    Suporta:
      1) /input/paciente.nii.gz
      2) /input/paciente/paciente.nii.gz
      3) /input/paciente/qualquer_nome.nii.gz, desde que haja apenas um .nii.gz na pasta.

    Retorna uma lista de tuplas: (subject, caminho_da_imagem_original).
    """
    subjects = {}

    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)

        if os.path.isfile(item_path) and item.endswith(".nii.gz"):
            subject = nifti_stem(item)
            if subject in subjects:
                raise RuntimeError(f"Subject duplicado detectado: {subject}")
            subjects[subject] = item_path
            continue

        if os.path.isdir(item_path):
            expected = os.path.join(item_path, f"{item}.nii.gz")
            if os.path.isfile(expected):
                subject = item
                if subject in subjects:
                    raise RuntimeError(f"Subject duplicado detectado: {subject}")
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
                    raise RuntimeError(f"Subject duplicado detectado: {subject}")
                subjects[subject] = candidates[0]

    return sorted(subjects.items())


def select_subjects(subjects, requested_subject=None):
    """
    Filtra a lista descoberta quando --subject é informado.

    Aceita o identificador do paciente com ou sem a extensão .nii.gz.
    Sem requested_subject, retorna todos os pacientes.
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
            f"Subject/imagem '{requested_subject}' não encontrado em INPUT_DIR. "
            f"Disponíveis: {available or 'nenhum'}"
        )

    if len(matches) > 1:
        matched_names = ", ".join(subject for subject, _ in matches)
        raise ValueError(
            f"A referência '{requested_subject}' é ambígua. Correspondências: {matched_names}"
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
    """Copia a imagem original para um workspace interno com o nome esperado pelo pipeline."""
    os.makedirs(destination_dir, exist_ok=True)
    destination = os.path.join(destination_dir, f"{subject}.nii.gz")
    shutil.copy2(input_file, destination)
    return destination


def run_pipeline(input_dir, output_dir, requested_subject=None):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isdir(input_dir):
        print(f"[-] Diretório de entrada não encontrado: {input_dir}")
        sys.exit(1)

    # Proteção contra configuração acidental que voltaria a misturar entrada e saída.
    real_input = os.path.realpath(input_dir)
    real_output = os.path.realpath(output_dir)
    if real_input == real_output or os.path.commonpath([real_input, real_output]) == real_input:
        print("[-] OUTPUT_DIR não pode ser igual nem ficar dentro de INPUT_DIR.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print("\n[🚀] Iniciando Pipeline Automatizada Enigma-SC (Passos 1, 2, 3, 5, 7)")
    print(f"[📥] Entrada : {input_dir}")
    print(f"[📤] Saída   : {output_dir}")

    # ---------------------------------------------------------
    # PASSO 1: Descoberta dos arquivos SEM alterar a entrada
    # ---------------------------------------------------------
    print("\n[1/5] Verificando arquivos de entrada...")
    subjects = discover_subjects(input_dir)

    if not subjects:
        print(f"[-] Nenhum dado válido (.nii.gz) encontrado em {input_dir}. Encerrando.")
        sys.exit(0)

    try:
        subjects = select_subjects(subjects, requested_subject)
    except ValueError as exc:
        print(f"[-] {exc}")
        sys.exit(1)

    if requested_subject:
        print(f"[🎯] Modo individual: {subjects[0][0]}")
    else:
        print(f"[📚] Modo batch: todas as {len(subjects)} imagem(ns) serão processadas.")

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
        print(f"🧠 Processando paciente: {subject}")
        print(f"📄 Original: {input_file}")
        print(f"📁 Resultados: {subject_output_dir}")
        print("=========================================")

        # PASSO 2: Segmentação da Medula
        print("[2/5] Segmentação da Medula...")
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

        # PASSO 3: Labeling Vertebral Automático
        print("[3/5] Labeling Vertebral Automático...")
        os.makedirs(NNUNET_IMAGES_TS, exist_ok=True)
        os.makedirs(NNUNET_PREDS, exist_ok=True)

        dest_label = os.path.join(NNUNET_IMAGES_TS, f"{subject}_0000.nii.gz")
        pred_file = os.path.join(NNUNET_PREDS, f"{subject}.nii.gz")

        # Evita reutilizar uma predição antiga em caso de falha silenciosa.
        if os.path.exists(pred_file):
            os.remove(pred_file)

        shutil.copy2(input_file, dest_label)

        if subprocess.run("python3 /home/scripts/cuda.py", shell=True).returncode != 0:
            print("      ⚠️ GPU não disponível para esta tarefa. Executando via CPU (mais lento)...")
            subprocess.run("python3 /home/scripts/cpu.py", shell=True, check=True)

        shutil.copy2(
            pred_file,
            os.path.join(subject_output_dir, f"{subject}_seg_labeled.nii.gz"),
        )

        if os.path.exists(dest_label):
            os.remove(dest_label)
        if os.path.exists(pred_file):
            os.remove(pred_file)

        # PASSO 5: Registro Automático
        print("[4/5] Registro Automático da Medula...")
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
            print("      ⚠️ Aviso: Alguns arquivos de registro não foram gerados corretamente.")

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

        # PASSO 7: Extração e Cálculos de Dados
        print("[5/5] Extração de Dados Clínicos...")
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

    print(f"\n[✅] Pipeline concluída! Resultados disponíveis em: {output_dir}")
    print(f"[🔒] Diretório de entrada preservado: {input_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.subject)
