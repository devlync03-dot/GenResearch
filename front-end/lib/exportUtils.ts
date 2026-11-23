import { jsPDF } from "jspdf";
import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType } from "docx";

function cleanMarkdownInline(text: string) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/__(.*?)__/g, "$1")
    .replace(/`([^`]+)`/g, "$1");
}

/**
 * Enhanced PDF export with page numbers, headers, footers, and academic formatting.
 */
export function exportToPDF(title: string, text: string) {
  const doc = new jsPDF();
  const pageHeight = doc.internal.pageSize.height;
  const pageWidth = doc.internal.pageSize.width;
  const margin = 20;
  const maxLineWidth = pageWidth - (margin * 2);

  // Header
  doc.setFont("times", "bold");
  doc.setTextColor(0, 0, 0);
  doc.setFontSize(16);
  doc.text(title, margin, margin + 5);

  doc.setLineWidth(0.5);
  doc.setDrawColor(0, 0, 0);
  doc.line(margin, margin + 9, pageWidth - margin, margin + 9);

  // Body
  doc.setFont("times", "normal");
  doc.setFontSize(12);

  const lines = text.split("\n").map((line) => line.trim());
  let y = margin + 20;

  const checkAddPage = (neededSpace = 10) => {
    if (y + neededSpace > pageHeight - margin) {
      doc.addPage();
      y = margin + 10;
    }
  };

  for (const line of lines) {
    if (line.startsWith("### ")) {
      checkAddPage(12);
      doc.setFont("times", "bold");
      doc.setFontSize(14);
      doc.text(cleanMarkdownInline(line.replace(/^#{1,4}\s+/, "")), margin, y);
      y += 8;
      doc.setFont("times", "normal");
      doc.setFontSize(12);
    } else if (line.startsWith("## ")) {
      checkAddPage(14);
      doc.setFont("times", "bold");
      doc.setFontSize(14);
      doc.text(cleanMarkdownInline(line.replace(/^#{1,4}\s+/, "")), margin, y);
      y += 10;
      doc.setFont("times", "normal");
      doc.setFontSize(12);
    } else if (line.startsWith("# ")) {
      checkAddPage(16);
      doc.setFont("times", "bold");
      doc.setFontSize(16);
      doc.text(cleanMarkdownInline(line.replace(/^#{1,4}\s+/, "")), margin, y);
      y += 12;
      doc.setFont("times", "normal");
      doc.setFontSize(12);
    } else if (line.trim().length > 0) {
      const cleanLine = cleanMarkdownInline(line).replace(/^[-*+]\s+/, "• ");
      const splitText = doc.splitTextToSize(cleanLine, maxLineWidth);
      checkAddPage(splitText.length * 6.35);
      doc.text(splitText, margin, y, { align: "justify", maxWidth: maxLineWidth });
      y += (splitText.length * 6.35);
    } else {
      y += 4;
    }
  }

  // Footer / Page numbers
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    doc.setFont("times", "italic");
    doc.setFontSize(9);
    doc.setTextColor(0, 0, 0);
    doc.text(`GenResearch Academic System  |  Page ${i} of ${totalPages}`, pageWidth / 2, pageHeight - 10, { align: "center" });
  }

  doc.save(`${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.pdf`);
}

/**
 * Enhanced DOCX export with professional academic typography and heading styles.
 */
export async function exportToDOCX(title: string, text: string) {
  const lines = text.split("\n").map((line) => line.trim());
  const children: Paragraph[] = [];

  // Title Header
  children.push(
    new Paragraph({
      children: [new TextRun({ text: title, font: "Arial", size: 32, bold: true })],
      alignment: AlignmentType.JUSTIFIED,
      spacing: { line: 360, after: 300 },
    })
  );

  for (const line of lines) {
    if (line.startsWith("### ")) {
      children.push(
        new Paragraph({
          children: [new TextRun({ text: line.replace("### ", ""), font: "Arial", size: 28, bold: true, color: "000000" })],
          heading: HeadingLevel.HEADING_3,
          alignment: AlignmentType.JUSTIFIED,
          spacing: { line: 360, before: 200, after: 100 },
        })
      );
    } else if (line.startsWith("## ")) {
      children.push(
        new Paragraph({
          children: [new TextRun({ text: line.replace("## ", ""), font: "Arial", size: 28, bold: true, color: "000000" })],
          heading: HeadingLevel.HEADING_2,
          alignment: AlignmentType.JUSTIFIED,
          spacing: { line: 360, before: 300, after: 150 },
        })
      );
    } else if (line.startsWith("# ")) {
      children.push(
        new Paragraph({
          children: [new TextRun({ text: line.replace("# ", ""), font: "Arial", size: 32, bold: true, color: "000000" })],
          heading: HeadingLevel.HEADING_1,
          alignment: AlignmentType.JUSTIFIED,
          spacing: { line: 360, before: 400, after: 200 },
        })
      );
    } else if (line.trim().length > 0) {
      const listItem = line.match(/^[-*+]\s+(.+)$/);
      children.push(
        new Paragraph({
          children: [new TextRun({ text: cleanMarkdownInline(listItem ? `• ${listItem[1]}` : line), font: "Arial", size: 24, color: "000000" })],
          alignment: AlignmentType.JUSTIFIED,
          spacing: { line: 360, after: 120 },
        })
      );
    } else {
      children.push(new Paragraph({ text: "", spacing: { after: 100 } }));
    }
  }

  const doc = new Document({
    sections: [{
      properties: {},
      children: children,
    }]
  });

  const blob = await Packer.toBlob(doc);
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.docx`;
  a.click();
  window.URL.revokeObjectURL(url);
}
