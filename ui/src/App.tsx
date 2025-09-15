import React, { useRef, useState } from "react";
import Canvas from "./components/Canvas";
import { API_URL } from "./config";

type Prediction = {
  digit: number;
  confidence: number;
};

const CANVAS_SIZE = 196;

const App: React.FC = () => {
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleExport = async (blob: Blob) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData();
      form.append("file", blob, "digit.png");
      const resp = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: form,
      });
      if (!resp.ok) {
        throw new Error(`API error: ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (e: any) {
      setError(e.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>MNIST Digit Classifier</h1>
      <Canvas
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onExport={handleExport}
      />
      <div className="buttons-row">
        <button
          type="button"
          onClick={() => {
            // Find the canvas and export
            const canvas = document.querySelector("canvas");
            if (canvas) {
              canvas.toBlob((blob) => {
                if (blob) handleExport(blob);
              }, "image/png");
            }
          }}
          disabled={loading}
        >
          Predict
        </button>
        <button
          type="button"
          onClick={() => {
            const canvas = document.querySelector("canvas");
            if (canvas) {
              const ctx = (canvas as HTMLCanvasElement).getContext("2d")!;
              ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
              ctx.fillStyle = "black";
              ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
            }
            setResult(null);
            setError(null);
          }}
        >
          Clear
        </button>
      </div>
      <div className="result-panel">
        {loading && <div>Predicting...</div>}
        {error && <div className="error">Error: {error}</div>}
        {result && (
          <div>
            <div>
              <b>Digit:</b> {result.digit}
            </div>
            <div>
              <b>Confidence:</b> {(result.confidence * 100).toFixed(2)}%
            </div>
          </div>
        )}
      </div>
      <footer>
        <small>
          Draw a digit (0-9) and click Predict.<br />
          API: <code>{API_URL}</code>
        </small>
      </footer>
    </div>
  );
};

export default App;
