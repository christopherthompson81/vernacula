using System.Text.Json;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;

namespace ParakeetCSharp.Services;

internal class ExportService
{
    // ── Excel ────────────────────────────────────────────────────────────────

    public void ExportXlsx(TranscriptionDb db, string outputPath)
    {
        var rows = db.GetTranscript();
        AudioUtils.WriteXlsx(rows, outputPath);
    }

    // ── CSV ──────────────────────────────────────────────────────────────────

    public void ExportCsv(TranscriptionDb db, string outputPath)
    {
        var rows = db.GetTranscript();
        using var writer = new StreamWriter(outputPath, append: false, Encoding.UTF8);
        writer.WriteLine("speaker_name,start_time,end_time,content");
        foreach (var row in rows)
        {
            string speaker = CsvEscape(row.GetValueOrDefault("speaker_name", ""));
            string start   = CsvEscape(row.GetValueOrDefault("start_time",   ""));
            string end     = CsvEscape(row.GetValueOrDefault("end_time",     ""));
            string content = CsvEscape(row.GetValueOrDefault("content",      ""));
            writer.WriteLine($"{speaker},{start},{end},{content}");
        }
    }

    // ── JSON ─────────────────────────────────────────────────────────────────

    public void ExportJson(TranscriptionDb db, string outputPath)
    {
        var rows = db.GetTranscript();
        var items = rows.Select(r => new
        {
            speaker = r.GetValueOrDefault("speaker_name", ""),
            start   = r.GetValueOrDefault("start_time",   ""),
            end     = r.GetValueOrDefault("end_time",     ""),
            content = r.GetValueOrDefault("content",      ""),
        }).ToList();
        var json = JsonSerializer.Serialize(items, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(outputPath, json, Encoding.UTF8);
    }

    // ── SRT ──────────────────────────────────────────────────────────────────

    public void ExportSrt(TranscriptionDb db, string outputPath)
    {
        var rows = db.GetTranscriptRows();
        using var writer = new StreamWriter(outputPath, append: false, Encoding.UTF8);
        int index = 1;
        foreach (var (speaker, startSec, endSec, content) in rows)
        {
            writer.WriteLine(index++);
            writer.WriteLine($"{ToSrtTime(startSec)} --> {ToSrtTime(endSec)}");
            writer.WriteLine($"{speaker}: {content}");
            writer.WriteLine();
        }
    }

    // ── Markdown ─────────────────────────────────────────────────────────────

    public void ExportMd(TranscriptionDb db, string outputPath)
    {
        var rows = db.GetTranscriptRows();
        using var writer = new StreamWriter(outputPath, append: false, Encoding.UTF8);
        writer.WriteLine("# Transcript");
        writer.WriteLine();
        foreach (var (speaker, startSec, endSec, content) in rows)
        {
            string start = AudioUtils.SecondsToHhMmSs(startSec);
            string end   = AudioUtils.SecondsToHhMmSs(endSec);
            writer.WriteLine($"**{speaker}** `[{start} → {end}]`");
            writer.WriteLine();
            writer.WriteLine(content);
            writer.WriteLine();
            writer.WriteLine("---");
            writer.WriteLine();
        }
    }

    // ── DOCX ─────────────────────────────────────────────────────────────────

    public void ExportDocx(TranscriptionDb db, string outputPath)
    {
        var rows = db.GetTranscriptRows();

        using var doc = WordprocessingDocument.Create(outputPath, WordprocessingDocumentType.Document);
        var mainPart = doc.AddMainDocumentPart();
        var body = new Body();
        mainPart.Document = new Document(body);

        // Title
        body.Append(new Paragraph(
            new Run(
                new RunProperties(new Bold(), new FontSize { Val = "36" }),
                new Text("Transcript"))));
        body.Append(new Paragraph());

        foreach (var (speaker, startSec, endSec, content) in rows)
        {
            string start = AudioUtils.SecondsToHhMmSs(startSec);
            string end   = AudioUtils.SecondsToHhMmSs(endSec);

            // Speaker + time heading (bold)
            body.Append(new Paragraph(
                new Run(
                    new RunProperties(new Bold()),
                    new Text($"[{start} \u2013 {end}] {speaker}"))));

            // Content
            body.Append(new Paragraph(
                new Run(
                    new Text(content) { Space = SpaceProcessingModeValues.Preserve })));

            // Spacing paragraph
            body.Append(new Paragraph());
        }

        mainPart.Document.Save();
    }

    // ── SQLite DB copy ────────────────────────────────────────────────────────

    public void ExportDbCopy(string sourceDbPath, string destPath)
        => File.Copy(sourceDbPath, destPath, overwrite: true);

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static string CsvEscape(string s)
    {
        if (s.Contains(',') || s.Contains('"') || s.Contains('\n'))
            return $"\"{s.Replace("\"", "\"\"")}\"";
        return s;
    }

    private static string ToSrtTime(double seconds)
    {
        var ts = TimeSpan.FromSeconds(seconds);
        return $"{(int)ts.TotalHours:D2}:{ts.Minutes:D2}:{ts.Seconds:D2},{ts.Milliseconds:D3}";
    }
}
