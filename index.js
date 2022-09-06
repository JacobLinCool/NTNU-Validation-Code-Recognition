const os = require("node:os");
const fs = require("node:fs");
const path = require("node:path");
const { exec } = require("node:child_process");
const express = require("express");
const multer = require("multer");

const dir = path.join(os.tmpdir(), "uploads");
fs.mkdirSync(dir, { recursive: true });

const server = express();
const upload = multer({ dest: dir });

server.post("/recognize", upload.single("image"), async (req, res) => {
    const file = req.file;
    if (!file) {
        res.status(400).json({ error: "No files were uploaded." });
        return;
    }

    try {
        const program = exec(`python3 rec/predict.py --image ${file.path}`);
        const result = await new Promise((resolve, reject) => {
            let data = "";
            let error = "";
            program.stdout.on("data", (buf) => {
                data += buf.toString();
            });
            program.stderr.on("data", (buf) => {
                error += buf.toString();
            });
            program.on("close", () => {
                const [, result] = data.split("\n");
                if (result) {
                    if (result[3] === "=") {
                        const answer = new Function(`return ${result.slice(0, 3)}`)();
                        resolve(answer.toString());
                    } else {
                        resolve(result);
                    }
                } else {
                    reject(error);
                }
            });
        });

        res.json({ result });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message });
    } finally {
        if (fs.existsSync(file.path)) {
            fs.unlinkSync(file.path);
        }
    }
});

const port = Number(process.env.PORT) || 3000;
server.listen(port, () => {
    console.log(`Server started on port ${port}`);
});
