// server.js
import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const app = express();

// __dirname is not available in ES modules, so we derive it
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, "public")));

// Serve index.html at the root URL
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(
    `Server is running on port ${PORT}. Visit http://localhost:${PORT} in your browser.`
  );
});
