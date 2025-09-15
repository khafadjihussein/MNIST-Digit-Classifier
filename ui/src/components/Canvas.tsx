import React, { useRef, useEffect, useState } from "react";

type CanvasProps = {
  width: number;
  height: number;
  onExport: (blob: Blob) => void;
};

const Canvas: React.FC<CanvasProps> = ({ width, height, onExport }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [drawing, setDrawing] = useState(false);

  useEffect(() => {
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, width, height);
  }, [width, height]);

  const getPos = (e: React.MouseEvent | React.TouchEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    if ("touches" in e) {
      const t = e.touches[0];
      return { x: t.clientX - rect.left, y: t.clientY - rect.top };
    } else {
      return { x: (e as React.MouseEvent).clientX - rect.left, y: (e as React.MouseEvent).clientY - rect.top };
    }
  };

  const startDraw = (e: React.MouseEvent | React.TouchEvent) => {
    setDrawing(true);
  };

  const endDraw = () => setDrawing(false);

  const draw = (e: React.MouseEvent | React.TouchEvent) => {
    if (!drawing) return;
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 12;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalCompositeOperation = "source-over";
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.beginPath();
    const { x, y } = getPos(e);
    ctx.moveTo(x, y);
    startDraw(e);
  };

  const handleMouseUp = (e: React.MouseEvent) => {
    endDraw();
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.beginPath();
  };

  const handleMouseMove = (e: React.MouseEvent) => draw(e);

  const handleTouchStart = (e: React.TouchEvent) => {
    e.preventDefault();
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.beginPath();
    const { x, y } = getPos(e);
    ctx.moveTo(x, y);
    startDraw(e);
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    endDraw();
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.beginPath();
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    e.preventDefault();
    draw(e);
  };

  const handleClear = () => {
    const ctx = canvasRef.current!.getContext("2d")!;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, width, height);
  };

  const handleExport = async () => {
    if (canvasRef.current) {
      canvasRef.current.toBlob((blob) => {
        if (blob) onExport(blob);
      }, "image/png");
    }
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ background: "black", touchAction: "none" }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseOut={endDraw}
        onMouseMove={handleMouseMove}
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
        onTouchCancel={endDraw}
        onTouchMove={handleTouchMove}
      />
      <div className="canvas-buttons">
        <button type="button" onClick={handleClear}>Clear</button>
        <button type="button" onClick={handleExport}>Export</button>
      </div>
    </div>
  );
};

export default Canvas;
