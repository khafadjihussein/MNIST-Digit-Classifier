/**
 * Export a canvas as a PNG Blob for multipart/form-data.
 * Optionally resize to 28x28 for preview.
 */
export async function canvasToBlob(
  canvas: HTMLCanvasElement,
  size: number = 28
): Promise<Blob> {
  // Resize to 28x28 for preview, but send original for API
  if (canvas.width === size && canvas.height === size) {
    return await new Promise((resolve) =>
      canvas.toBlob((blob) => resolve(blob!), "image/png")
    );
  }
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  const ctx = tmp.getContext("2d")!;
  ctx.drawImage(canvas, 0, 0, size, size);
  return await new Promise((resolve) =>
    tmp.toBlob((blob) => resolve(blob!), "image/png")
  );
}
