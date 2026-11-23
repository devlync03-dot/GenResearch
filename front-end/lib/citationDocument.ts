const MAX_FILE_BYTES = 12 * 1024 * 1024;
const MAX_TEXT_CHARS = 500_000;

export async function extractCitationDocumentText(file: File): Promise<string> {
  if (file.size > MAX_FILE_BYTES) {
    throw new Error("Document exceeds the 12 MB limit.");
  }

  const extension = file.name.split(".").pop()?.toLowerCase();
  let text: string;

  if (extension === "txt") {
    const bytes = await file.arrayBuffer();
    try {
      text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch {
      throw new Error("TXT files must use UTF-8 encoding.");
    }
  } else if (extension === "docx") {
    const mammoth = (await import("mammoth")).default;
    const extracted = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() });
    text = extracted.value;
  } else if (extension === "pdf") {
    const pdfjs = await import("pdfjs-dist/legacy/build/pdf.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = new URL(
      "pdfjs-dist/legacy/build/pdf.worker.min.mjs",
      import.meta.url,
    ).toString();
    const pdf = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;
    const pages: string[] = [];
    let textLength = 0;

    try {
      for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
        const page = await pdf.getPage(pageNumber);
        const content = await page.getTextContent();
        const pageText = content.items
          .map(item => "str" in item ? `${item.str}${item.hasEOL ? "\n" : " "}` : "")
          .join("");
        pages.push(pageText);
        textLength += pageText.length;
        if (textLength > MAX_TEXT_CHARS) {
          throw new Error("Extracted document text exceeds the 500,000-character limit.");
        }
      }
    } finally {
      await pdf.destroy();
    }

    text = pages.join("\n");
  } else {
    throw new Error("Choose a PDF, DOCX, or UTF-8 TXT document.");
  }

  const normalizedText = text.replace(/\u0000/g, "").trim();
  if (!normalizedText) {
    throw new Error("No readable text was extracted. Scanned PDFs need OCR before audit.");
  }
  if (normalizedText.length > MAX_TEXT_CHARS) {
    throw new Error("Extracted document text exceeds the 500,000-character limit.");
  }

  return normalizedText;
}
