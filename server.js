const express = require("express");
const multer = require("multer");
const fs = require("fs");
const path = require("path");
const os = require("os");
const crypto = require("crypto");
const readline = require("readline");
const { spawn } = require("child_process");

const PORT = process.env.PORT || 3000;
const MODEL_PATH = path.join(__dirname, "models_saved", "model.pth");
const PREPROCESS_MODE = "manga_norm";
const INFERENCE_TIMEOUT_MS = Number(process.env.INFERENCE_TIMEOUT_MS || 30000);
const PYTHON_CMD = process.env.PYTHON_CMD || "uv";
const PYTHON_ARGS = process.env.PYTHON_ARGS
    ? process.env.PYTHON_ARGS.split(" ").filter(Boolean)
    : ["run", "python"];

const RUNTIME_DIR = path.join(__dirname, ".runtime");
const UPLOAD_DIR = path.join(RUNTIME_DIR, "uploads");
const WORKER_PATH = path.join(RUNTIME_DIR, "inference_worker.py");
const PUBLIC_DIR = path.join(__dirname, "public");

fs.mkdirSync(RUNTIME_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const PYTHON_WORKER_CODE = String.raw`
import sys
import json
import argparse
import traceback

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps, ImageFilter, ImageEnhance


PREPROCESS_MODE = "manga_norm"


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


def print_json(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def model_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def validate_state_dict(state_dict):
    expected_key = "network.22.weight"

    if expected_key not in state_dict:
        sample_keys = list(state_dict.keys())[:12]
        raise ValueError(
            "Checkpoint não é compatível com a CNN adaptativa atual. "
            f"Primeiras chaves do state_dict: {sample_keys}"
        )

    weight_shape = tuple(state_dict[expected_key].shape)

    if weight_shape != (128, 256):
        raise ValueError(
            "Checkpoint não é compatível com a CNN adaptativa atual. "
            f"Esperado {expected_key} com shape (128, 256), recebido {weight_shape}."
        )


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        classes = checkpoint.get("classes", [])
        config = checkpoint.get("config", {})
    else:
        classes = []
        config = {}

    state_dict = model_state_dict(checkpoint)
    validate_state_dict(state_dict)

    model = CNN().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    resize_size = int(config.get("resize_size", 256))
    crop_size = int(config.get("image_size", 224))

    transform = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    return {
        "name": model_path.split("/")[-1],
        "path": model_path,
        "model": model,
        "classes": classes,
        "config": config,
        "device": device,
        "transform": transform,
        "architecture": "adaptive_224"
    }


def open_transposed_image(image_path):
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).copy()


def normalize_manga_image(image):
    image = image.convert("L")
    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return image.convert("RGB")


def prepare_image(image_path):
    image = open_transposed_image(image_path)
    return normalize_manga_image(image)


def predict_image(model_info, image_path):
    model = model_info["model"]
    classes = model_info["classes"]
    device = model_info["device"]
    transform = model_info["transform"]
    image = prepare_image(image_path)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())

    if classes and predicted_index < len(classes):
        predicted_class = classes[predicted_index]
    else:
        predicted_class = f"classe_indice_{predicted_index}_sem_nome_no_checkpoint"

    top_k = min(3, probabilities.shape[0])
    top_values, top_indices = torch.topk(probabilities, k=top_k)
    top_predictions = []

    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        index = int(index)

        if classes and index < len(classes):
            class_name = classes[index]
        else:
            class_name = f"classe_indice_{index}_sem_nome_no_checkpoint"

        top_predictions.append({
            "class": class_name,
            "index": index,
            "confidence": float(value)
        })

    return {
        "modelName": model_info["name"],
        "modelPath": model_info["path"],
        "architecture": model_info["architecture"],
        "preprocessing": PREPROCESS_MODE,
        "predictedClass": predicted_class,
        "predictedIndex": predicted_index,
        "confidence": confidence,
        "topPredictions": top_predictions
    }


def public_model_metadata(model_info):
    return {
        "name": model_info["name"],
        "path": model_info["path"],
        "architecture": model_info["architecture"],
        "classes": model_info["classes"],
        "config": model_info["config"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        model_info = load_model(args.model)

        print_json({
            "status": "ready",
            "message": "Modelo carregado com sucesso.",
            "device": model_info["device"],
            "preprocessMode": PREPROCESS_MODE,
            "model": public_model_metadata(model_info)
        })

    except Exception as e:
        print_json({
            "status": "startup_error",
            "message": str(e),
            "traceback": traceback.format_exc()
        })
        sys.exit(1)

    for line in sys.stdin:
        try:
            request = json.loads(line)

            request_id = request.get("id")
            image_path = request.get("imagePath")

            if not image_path:
                raise ValueError("Campo imagePath não foi informado.")

            result = predict_image(model_info, image_path)

            print_json({
                "id": request_id,
                "ok": True,
                "result": result
            })

        except Exception as e:
            print_json({
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            })


if __name__ == "__main__":
    main()
`;

fs.writeFileSync(WORKER_PATH, PYTHON_WORKER_CODE, "utf8");

function validateModelFile() {
    if (!fs.existsSync(MODEL_PATH)) {
        console.error("\nERRO: arquivo model.pth não encontrado.");
        console.error(`Caminho esperado: ${MODEL_PATH}`);
        console.error("\nCopie seu modelo treinado para:");
        console.error("models_saved/model.pth");
        process.exit(1);
    }
}

let workerProcess = null;
let workerReady = false;
let pendingRequests = new Map();

function startPythonWorker() {
    return new Promise((resolve, reject) => {
        workerProcess = spawn(PYTHON_CMD, [
            ...PYTHON_ARGS,
            WORKER_PATH,
            "--model",
            MODEL_PATH
        ]);

        const rl = readline.createInterface({
            input: workerProcess.stdout
        });

        const startupTimeout = setTimeout(() => {
            reject(new Error("Timeout ao tentar carregar o modelo PyTorch."));
        }, 30000);

        rl.on("line", (line) => {
            let message;

            try {
                message = JSON.parse(line);
            } catch (error) {
                console.error("Saída inválida do worker Python:", line);
                return;
            }

            if (message.status === "ready") {
                clearTimeout(startupTimeout);
                workerReady = true;

                console.log("\nAPI iniciada com modelo carregado.");
                console.log(`Dispositivo usado pelo PyTorch: ${message.device}`);
                console.log(`Modelo carregado: ${message.model.name} (${message.model.architecture})`);
                console.log(`Pré-processamento: ${message.preprocessMode}`);

                resolve(message);
                return;
            }

            if (message.status === "startup_error") {
                clearTimeout(startupTimeout);

                console.error("\nERRO ao carregar o modelo:");
                console.error(message.message);
                console.error(message.traceback);

                reject(new Error(message.message));
                return;
            }

            if (message.id && pendingRequests.has(message.id)) {
                const { resolve, reject, timeout } = pendingRequests.get(message.id);

                clearTimeout(timeout);
                pendingRequests.delete(message.id);

                if (message.ok) {
                    resolve(message.result);
                } else {
                    reject(new Error(message.error));
                }
            }
        });

        workerProcess.stderr.on("data", (data) => {
            console.error("[Python stderr]", data.toString());
        });

        workerProcess.on("exit", (code) => {
            workerReady = false;

            if (pendingRequests.size > 0) {
                for (const [, request] of pendingRequests.entries()) {
                    clearTimeout(request.timeout);
                    request.reject(new Error("Worker Python foi encerrado."));
                }

                pendingRequests.clear();
            }

            console.error(`Worker Python encerrado com código: ${code}`);
        });

        workerProcess.on("error", (error) => {
            clearTimeout(startupTimeout);
            reject(error);
        });
    });
}

function runInference(imagePath) {
    return new Promise((resolve, reject) => {
        if (!workerReady || !workerProcess) {
            reject(new Error("Modelo ainda não está pronto para inferência."));
            return;
        }

        const id = crypto.randomUUID();

        const timeout = setTimeout(() => {
            pendingRequests.delete(id);
            reject(new Error("Timeout durante a inferência."));
        }, INFERENCE_TIMEOUT_MS);

        pendingRequests.set(id, {
            resolve,
            reject,
            timeout
        });

        workerProcess.stdin.write(JSON.stringify({
            id,
            imagePath
        }) + os.EOL);
    });
}

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, UPLOAD_DIR);
    },
    filename: function (req, file, cb) {
        const extension = path.extname(file.originalname || "").toLowerCase();
        const safeExtension = extension || ".jpg";
        cb(null, `${crypto.randomUUID()}${safeExtension}`);
    }
});

const upload = multer({
    storage,
    limits: {
        fileSize: 8 * 1024 * 1024
    },
    fileFilter: function (req, file, cb) {
        const allowedMimeTypes = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp"
        ];

        if (!allowedMimeTypes.includes(file.mimetype)) {
            cb(new Error("Formato inválido. Envie uma imagem JPEG, PNG, WEBP ou BMP."));
            return;
        }

        cb(null, true);
    }
});

async function bootstrap() {
    console.log("Inicializando API de inferência CNN...");
    console.log(`Modelo esperado: ${MODEL_PATH}`);

    validateModelFile();

    try {
        await startPythonWorker();
    } catch (error) {
        console.error("\nNão foi possível iniciar a API.");
        console.error(error.message);
        process.exit(1);
    }

    const app = express();

    app.use(express.static(PUBLIC_DIR));

    app.post("/infer", upload.single("image"), async (req, res) => {
        if (!req.file) {
            return res.status(400).json({
                ok: false,
                error: "Nenhuma imagem foi enviada. Use o campo multipart chamado 'image'."
            });
        }

        const imagePath = req.file.path;

        try {
            const result = await runInference(imagePath);

            return res.json({
                ok: true,
                ...result
            });

        } catch (error) {
            return res.status(500).json({
                ok: false,
                error: error.message
            });

        } finally {
            fs.unlink(imagePath, () => { });
        }
    });

    app.use((error, req, res, next) => {
        return res.status(400).json({
            ok: false,
            error: error.message
        });
    });

    app.listen(PORT, () => {
        console.log(`\nServidor rodando em: http://localhost:${PORT}`);
        console.log(`Interface web: http://localhost:${PORT}`);
        console.log(`Endpoint de inferência: POST http://localhost:${PORT}/infer`);
        console.log("\nCampo esperado no multipart/form-data:");
        console.log("image");
    });
}

bootstrap();
