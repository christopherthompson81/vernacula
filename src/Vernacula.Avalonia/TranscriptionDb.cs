using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.Data.Sqlite;

namespace Vernacula.Avalonia;

/// <summary>
/// SQLite-backed persistence for diarization + ASR results.
/// C# port of sql.py / TranscriptionDB.
/// </summary>
internal sealed class TranscriptionDb : IDisposable
{
    private readonly SqliteConnection _conn;

    // ── DDL ───────────────────────────────────────────────────────────────────

    private const string CreateMetadata = """
        CREATE TABLE IF NOT EXISTS metadata (
            key   TEXT,
            value TEXT
        )
        """;

    private const string CreateResults = """
        CREATE TABLE IF NOT EXISTS results (
            result_id             INTEGER PRIMARY KEY AUTOINCREMENT,
            diarization_speaker_id INTEGER,
            speaker_id            INTEGER,
            start_time            NUMERIC,
            end_time              NUMERIC,
            start_time_f          TEXT,
            end_time_f            TEXT,
            asr_content           TEXT,
            content               TEXT,
            tokens                TEXT,
            timestamps            TEXT,
            logprobs              TEXT
        )
        """;

    private const string CreateSpeakers = """
        CREATE TABLE IF NOT EXISTS speakers (
            speaker_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT
        )
        """;

    private const string CreateTranscriptView = """
        CREATE VIEW IF NOT EXISTS transcript AS
        SELECT
            s.name        AS speaker_name,
            r.start_time_f AS start_time,
            r.end_time_f   AS end_time,
            r.content
        FROM
            results  r,
            speakers s
        WHERE
            r.speaker_id = s.speaker_id
        """;

    private const string CreateSegmentCards = """
        CREATE TABLE IF NOT EXISTS segment_cards (
            card_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            sort_order REAL    NOT NULL,
            speaker_id INTEGER NOT NULL,
            play_start NUMERIC NOT NULL,
            play_end   NUMERIC NOT NULL,
            content    TEXT,
            verified   INTEGER NOT NULL DEFAULT 0,
            suppressed INTEGER NOT NULL DEFAULT 0
        )
        """;

    private const string CreateCardSources = """
        CREATE TABLE IF NOT EXISTS card_sources (
            card_id         INTEGER NOT NULL REFERENCES segment_cards(card_id),
            result_id       INTEGER NOT NULL,
            token_start     INTEGER NOT NULL DEFAULT 0,
            token_end       INTEGER,
            ts_frame_offset INTEGER NOT NULL DEFAULT 0,
            source_order    INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (card_id, result_id)
        )
        """;

    // ── Construction ─────────────────────────────────────────────────────────

    public TranscriptionDb(string databasePath)
    {
        _conn = new SqliteConnection($"Data Source={databasePath};Pooling=false");
        _conn.Open();

        Execute(CreateMetadata);
        Execute(CreateResults);
        Execute(CreateSpeakers);
        Execute(CreateTranscriptView);
        Execute(CreateSegmentCards);
        Execute(CreateCardSources);

        // Lazy migrations for results columns
        try { Execute("ALTER TABLE results ADD COLUMN verified   INTEGER NOT NULL DEFAULT 0"); }
        catch (SqliteException) { /* column already exists */ }
        try { Execute("ALTER TABLE results ADD COLUMN suppressed INTEGER NOT NULL DEFAULT 0"); }
        catch (SqliteException) { /* column already exists */ }

        MigrateToCardLayer();
    }

    /// <summary>
    /// Idempotent: if segment_cards is empty but results has rows, populate card tables from results.
    /// Called automatically on open so existing DBs are upgraded transparently.
    /// </summary>
    private void MigrateToCardLayer()
    {
        using var checkCards = _conn.CreateCommand();
        checkCards.CommandText = "SELECT count(*) FROM segment_cards";
        long cardCount = (long)(checkCards.ExecuteScalar() ?? 0L);
        if (cardCount > 0) return;

        using var checkResults = _conn.CreateCommand();
        checkResults.CommandText = "SELECT count(*) FROM results";
        long resultCount = (long)(checkResults.ExecuteScalar() ?? 0L);
        if (resultCount == 0) return;

        Execute("""
            INSERT INTO segment_cards (card_id, sort_order, speaker_id, play_start, play_end,
                                       content, verified, suppressed)
            SELECT result_id,
                   CAST(result_id AS REAL),
                   speaker_id, start_time, end_time,
                   CASE WHEN content IS NOT NULL AND content != coalesce(asr_content,'')
                        THEN content ELSE NULL END,
                   coalesce(verified,   0),
                   coalesce(suppressed, 0)
            FROM results
            ORDER BY start_time, result_id
            """);

        Execute("""
            INSERT INTO card_sources (card_id, result_id, token_start, ts_frame_offset, source_order)
            SELECT result_id, result_id, 0, 0, 0
            FROM results
            ORDER BY result_id
            """);
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    public void InsertMetadata(string key, string value)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "INSERT INTO metadata(key, value) VALUES ($key, $val)";
        cmd.Parameters.AddWithValue("$key", key);
        cmd.Parameters.AddWithValue("$val", value);
        cmd.ExecuteNonQuery();
    }

    public void UpdateMetadata(string key, string value)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE metadata SET value = $val WHERE key = $key";
        cmd.Parameters.AddWithValue("$key", key);
        cmd.Parameters.AddWithValue("$val", value);
        cmd.ExecuteNonQuery();
    }

    public string? GetMetadata(string key)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT value FROM metadata WHERE key = $key";
        cmd.Parameters.AddWithValue("$key", key);
        return cmd.ExecuteScalar() as string;
    }

    /// <summary>Mirrors sql.py populate_metadata().</summary>
    public void PopulateMetadata(string audioPath)
    {
        InsertMetadata("audio_file",          audioPath);
        InsertMetadata("audio_file_sha256sum", AudioUtils.Sha256Checksum(audioPath));
        InsertMetadata("transcription_run_datestamp",
            DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
        InsertMetadata("file_datestamp",
            File.GetLastWriteTime(audioPath).ToString("yyyy-MM-dd HH:mm:ss"));
        InsertMetadata("diarization_model", "nvidia/diar_streaming_sortformer_4spk-v2.1");
        InsertMetadata("asr_model",         "nvidia/parakeet-tdt-0.6b-v3");
    }

    // ── Results (raw ASR data — never mutated by user edits) ──────────────────

    public void InsertResult(
        int     diarizationSpeakerId,
        int     speakerId,
        double  startTime,
        double  endTime,
        string  startTimeF,
        string  endTimeF,
        string? asrContent,
        string? content,
        string? tokens,
        string? timestamps,
        string? logprobs)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO results
                (diarization_speaker_id, speaker_id,
                 start_time, end_time, start_time_f, end_time_f,
                 asr_content, content, tokens, timestamps, logprobs)
            VALUES
                ($dspk, $spk, $st, $et, $stf, $etf, $asr, $con, $tok, $ts, $lp)
            """;
        cmd.Parameters.AddWithValue("$dspk", diarizationSpeakerId);
        cmd.Parameters.AddWithValue("$spk",  speakerId);
        cmd.Parameters.AddWithValue("$st",   startTime);
        cmd.Parameters.AddWithValue("$et",   endTime);
        cmd.Parameters.AddWithValue("$stf",  startTimeF);
        cmd.Parameters.AddWithValue("$etf",  endTimeF);
        cmd.Parameters.AddWithValue("$asr",  (object?)asrContent  ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$con",  (object?)content     ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$tok",  (object?)tokens      ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$ts",   (object?)timestamps  ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$lp",   (object?)logprobs    ?? DBNull.Value);
        cmd.ExecuteNonQuery();
    }

    public void UpdateResult(
        int     resultId,
        string? asrContent,
        string? content,
        string? tokens,
        string? timestamps,
        string? logprobs)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            UPDATE results
            SET asr_content = $asr,
                content     = $con,
                tokens      = $tok,
                timestamps  = $ts,
                logprobs    = $lp
            WHERE result_id = $id
            """;
        cmd.Parameters.AddWithValue("$asr", (object?)asrContent ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$con", (object?)content    ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$tok", (object?)tokens     ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$ts",  (object?)timestamps ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$lp",  (object?)logprobs   ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$id",  resultId);
        cmd.ExecuteNonQuery();
    }

    // ── Segment Cards (user-edit overlay) ─────────────────────────────────────

    /// <summary>Creates a new card row and returns its card_id.</summary>
    public int CreateCard(int speakerId, double playStart, double playEnd)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO segment_cards (sort_order, speaker_id, play_start, play_end)
            VALUES ((SELECT coalesce(max(sort_order), 0) + 1 FROM segment_cards),
                    $spk, $ps, $pe);
            SELECT last_insert_rowid();
            """;
        cmd.Parameters.AddWithValue("$spk", speakerId);
        cmd.Parameters.AddWithValue("$ps",  playStart);
        cmd.Parameters.AddWithValue("$pe",  playEnd);
        return Convert.ToInt32(cmd.ExecuteScalar());
    }

    /// <summary>Adds a source result to a card.</summary>
    public void AddCardSource(int cardId, int resultId,
        int tokenStart, int? tokenEnd, int tsFrameOffset, int sourceOrder)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO card_sources
                (card_id, result_id, token_start, token_end, ts_frame_offset, source_order)
            VALUES ($cid, $rid, $ts, $te, $tfo, $so)
            """;
        cmd.Parameters.AddWithValue("$cid", cardId);
        cmd.Parameters.AddWithValue("$rid", resultId);
        cmd.Parameters.AddWithValue("$ts",  tokenStart);
        cmd.Parameters.AddWithValue("$te",  tokenEnd.HasValue ? tokenEnd.Value : DBNull.Value);
        cmd.Parameters.AddWithValue("$tfo", tsFrameOffset);
        cmd.Parameters.AddWithValue("$so",  sourceOrder);
        cmd.ExecuteNonQuery();
    }

    /// <summary>Inserts a new result row for a re-transcribed segment and returns its result_id.</summary>
    public int InsertResultReturningId(
        int     speakerId,
        double  startTime,
        double  endTime,
        string? asrContent,
        string? tokens,
        string? timestamps,
        string? logprobs)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            INSERT INTO results
                (diarization_speaker_id, speaker_id, start_time, end_time,
                 start_time_f, end_time_f, asr_content, content, tokens, timestamps, logprobs)
            VALUES
                ($dspk, $spk, $st, $et, $stf, $etf, $asr, $asr, $tok, $ts, $lp);
            SELECT last_insert_rowid();
            """;
        cmd.Parameters.AddWithValue("$dspk", speakerId);
        cmd.Parameters.AddWithValue("$spk",  speakerId);
        cmd.Parameters.AddWithValue("$st",   startTime);
        cmd.Parameters.AddWithValue("$et",   endTime);
        cmd.Parameters.AddWithValue("$stf",  AudioUtils.SecondsToHhMmSs(Math.Round(startTime)));
        cmd.Parameters.AddWithValue("$etf",  AudioUtils.SecondsToHhMmSs(Math.Round(endTime)));
        cmd.Parameters.AddWithValue("$asr",  (object?)asrContent ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$tok",  (object?)tokens     ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$ts",   (object?)timestamps ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$lp",   (object?)logprobs   ?? DBNull.Value);
        return Convert.ToInt32(cmd.ExecuteScalar());
    }

    /// <summary>Removes all source mappings for a card without deleting the card itself.</summary>
    public void DeleteCardSources(int cardId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "DELETE FROM card_sources WHERE card_id = $id";
        cmd.Parameters.AddWithValue("$id", cardId);
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Deletes result rows from the given set that are no longer referenced by any card_source.
    /// Call with the old result_ids after replacing a card's sources.
    /// </summary>
    public void DeleteOrphanedResults(IEnumerable<int> resultIds)
    {
        foreach (int id in resultIds)
        {
            using var check = _conn.CreateCommand();
            check.CommandText = "SELECT count(*) FROM card_sources WHERE result_id = $id";
            check.Parameters.AddWithValue("$id", id);
            long count = (long)(check.ExecuteScalar() ?? 0L);
            if (count > 0) continue;

            using var del = _conn.CreateCommand();
            del.CommandText = "DELETE FROM results WHERE result_id = $id";
            del.Parameters.AddWithValue("$id", id);
            del.ExecuteNonQuery();
        }
    }

    /// <summary>Deletes a card and its source mappings.</summary>
    public void DeleteCard(int cardId)
    {
        using var cmd1 = _conn.CreateCommand();
        cmd1.CommandText = "DELETE FROM card_sources WHERE card_id = $id";
        cmd1.Parameters.AddWithValue("$id", cardId);
        cmd1.ExecuteNonQuery();

        using var cmd2 = _conn.CreateCommand();
        cmd2.CommandText = "DELETE FROM segment_cards WHERE card_id = $id";
        cmd2.Parameters.AddWithValue("$id", cardId);
        cmd2.ExecuteNonQuery();
    }

    public void UpdateCardContent(int cardId, string? content)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE segment_cards SET content = $c WHERE card_id = $id";
        cmd.Parameters.AddWithValue("$c",  (object?)content ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$id", cardId);
        cmd.ExecuteNonQuery();
    }

    public void UpdateCardTimes(int cardId, double playStart, double playEnd)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE segment_cards SET play_start = $ps, play_end = $pe WHERE card_id = $id";
        cmd.Parameters.AddWithValue("$ps", playStart);
        cmd.Parameters.AddWithValue("$pe", playEnd);
        cmd.Parameters.AddWithValue("$id", cardId);
        cmd.ExecuteNonQuery();
    }

    public void UpdateCardSpeaker(int cardId, int speakerId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE segment_cards SET speaker_id = $spk WHERE card_id = $id";
        cmd.Parameters.AddWithValue("$spk", speakerId);
        cmd.Parameters.AddWithValue("$id",  cardId);
        cmd.ExecuteNonQuery();
    }

    public void UpdateCardVerified(int cardId, bool verified)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE segment_cards SET verified = $v WHERE card_id = $id";
        cmd.Parameters.AddWithValue("$v",  verified ? 1 : 0);
        cmd.Parameters.AddWithValue("$id", cardId);
        cmd.ExecuteNonQuery();
    }

    public void UpdateCardSuppressed(int cardId, bool suppressed)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE segment_cards SET suppressed = $v WHERE card_id = $id";
        cmd.Parameters.AddWithValue("$v",  suppressed ? 1 : 0);
        cmd.Parameters.AddWithValue("$id", cardId);
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Renumbers sort_order for all cards to match the provided ordered list of card_ids.
    /// Call after any structural change (merge/split) to maintain stable sort order on reload.
    /// </summary>
    public void RenumberSortOrders(IEnumerable<int> cardIdsInOrder)
    {
        double order = 1.0;
        foreach (int cardId in cardIdsInOrder)
        {
            using var cmd = _conn.CreateCommand();
            cmd.CommandText = "UPDATE segment_cards SET sort_order = $so WHERE card_id = $id";
            cmd.Parameters.AddWithValue("$so", order++);
            cmd.Parameters.AddWithValue("$id", cardId);
            cmd.ExecuteNonQuery();
        }
    }

    /// <summary>Returns all cards with their aggregated source data for the transcript editor.</summary>
    public List<CardQueryRow> GetCardsForEditor()
    {
        // Raw per-source rows ordered by card play_start, then source_order
        var raw = new List<(
            int cardId, double sortOrder, int speakerId, string speakerName,
            double playStart, double playEnd, string? content, bool verified, bool suppressed,
            int resultId, int tokenStart, int? tokenEnd, int tsFrameOffset, int sourceOrder,
            string? tokensJson, string? timestampsJson, string? logprobsJson, string asrContent)>();

        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT c.card_id, c.sort_order, c.speaker_id, s.name,
                   c.play_start, c.play_end, c.content, c.verified, c.suppressed,
                   cs.result_id, cs.token_start, cs.token_end, cs.ts_frame_offset, cs.source_order,
                   r.tokens, r.timestamps, r.logprobs,
                   coalesce(r.asr_content, coalesce(r.content, ''))
            FROM segment_cards c
            JOIN speakers s  ON c.speaker_id  = s.speaker_id
            JOIN card_sources cs ON cs.card_id = c.card_id
            JOIN results r   ON r.result_id   = cs.result_id
            ORDER BY c.play_start, c.card_id, cs.source_order
            """;

        using (var reader = cmd.ExecuteReader())
        {
            while (reader.Read())
            {
                raw.Add((
                    reader.GetInt32(0),
                    reader.GetDouble(1),
                    reader.GetInt32(2),
                    reader.GetString(3),
                    reader.GetDouble(4),
                    reader.GetDouble(5),
                    reader.IsDBNull(6)  ? null : reader.GetString(6),
                    reader.GetInt32(7)  != 0,
                    reader.GetInt32(8)  != 0,
                    reader.GetInt32(9),
                    reader.GetInt32(10),
                    reader.IsDBNull(11) ? (int?)null : reader.GetInt32(11),
                    reader.GetInt32(12),
                    reader.GetInt32(13),
                    reader.IsDBNull(14) ? null : reader.GetString(14),
                    reader.IsDBNull(15) ? null : reader.GetString(15),
                    reader.IsDBNull(16) ? null : reader.GetString(16),
                    reader.GetString(17)));
            }
        }

        // Group by card_id and aggregate token arrays
        var cards = new List<CardQueryRow>();
        int i = 0;
        while (i < raw.Count)
        {
            var first  = raw[i];
            int cardId = first.cardId;

            var resultIds  = new List<int>();
            var sources    = new List<CardSource>();
            var tokens     = new List<int>();
            var timestamps = new List<int>();
            var logprobs   = new List<float>();
            var asrParts   = new List<string>();

            while (i < raw.Count && raw[i].cardId == cardId)
            {
                var row = raw[i++];
                resultIds.Add(row.resultId);
                sources.Add(new CardSource(row.resultId, row.tokenStart, row.tokenEnd,
                                           row.tsFrameOffset, row.sourceOrder));

                var srcTokens     = ParseJsonList<int>(row.tokensJson);
                var srcTimestamps = ParseJsonList<int>(row.timestampsJson);
                var srcLogprobs   = ParseJsonList<float>(row.logprobsJson);

                int tStart = row.tokenStart;
                int tEnd   = row.tokenEnd ?? srcTokens.Count;

                for (int t = tStart; t < tEnd && t < srcTokens.Count; t++)
                    tokens.Add(srcTokens[t]);

                for (int t = tStart; t < tEnd && t < srcTimestamps.Count; t++)
                    timestamps.Add(srcTimestamps[t] + row.tsFrameOffset);

                for (int t = tStart; t < tEnd && t < srcLogprobs.Count; t++)
                    logprobs.Add(srcLogprobs[t]);

                string part = row.asrContent.Trim();
                if (part.Length > 0) asrParts.Add(part);
            }

            string asrContent = string.Join(" ", asrParts);
            string displayContent = first.content ?? asrContent;

            cards.Add(new CardQueryRow(
                CardId:      cardId,
                SortOrder:   first.sortOrder,
                SpeakerId:   first.speakerId,
                SpeakerName: first.speakerName,
                PlayStart:   first.playStart,
                PlayEnd:     first.playEnd,
                Content:     displayContent,
                RawContent:  first.content,
                Verified:    first.verified,
                IsSuppressed: first.suppressed,
                Sources:     sources,
                Tokens:      tokens,
                Timestamps:  timestamps,
                Logprobs:    logprobs,
                AsrContent:  asrContent));
        }

        return cards;
    }

    // ── Speakers ──────────────────────────────────────────────────────────────

    public void InsertSpeaker(string name)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "INSERT INTO speakers(name) VALUES ($name)";
        cmd.Parameters.AddWithValue("$name", name);
        cmd.ExecuteNonQuery();
    }

    /// <summary>Inserts a new speaker row and returns its new speaker_id.</summary>
    public int AddSpeaker(string name)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "INSERT INTO speakers(name) VALUES ($name); SELECT last_insert_rowid();";
        cmd.Parameters.AddWithValue("$name", name);
        return Convert.ToInt32(cmd.ExecuteScalar());
    }

    public void UpdateSpeaker(int speakerId, string name)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE speakers SET name = $name WHERE speaker_id = $id";
        cmd.Parameters.AddWithValue("$name", name);
        cmd.Parameters.AddWithValue("$id",   speakerId);
        cmd.ExecuteNonQuery();
    }

    /// <summary>Returns all speakers as (SpeakerId, Name) pairs ordered by speaker_id.</summary>
    public List<(int SpeakerId, string Name)> GetSpeakers()
    {
        var result = new List<(int, string)>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT speaker_id, name FROM speakers ORDER BY speaker_id";
        using var r = cmd.ExecuteReader();
        while (r.Read())
            result.Add((r.GetInt32(0), r.GetString(1)));
        return result;
    }

    // ── Reads (segment_cards-aware) ────────────────────────────────────────────

    /// <summary>Returns ordered list of (startSec, endSec, speakerId) from the card overlay.</summary>
    public List<(double start, double end, string spkId)> GetSegments()
    {
        var segs = new List<(double, double, string)>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT c.play_start, c.play_end,
                   'speaker_' || (c.speaker_id - 1) AS speaker_id
            FROM segment_cards c
            WHERE c.suppressed = 0
            ORDER BY c.play_start, c.card_id
            """;
        using var r = cmd.ExecuteReader();
        while (r.Read())
            segs.Add((r.GetDouble(0), r.GetDouble(1), r.GetString(2)));
        return segs;
    }

    /// <summary>Returns ordered rows with raw numeric timestamps for export purposes.</summary>
    public List<(string speaker, double startSec, double endSec, string content)> GetTranscriptRows()
    {
        var rows = new List<(string, double, double, string)>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT s.name, c.play_start, c.play_end,
                   coalesce(c.content, coalesce(r.asr_content, ''))
            FROM segment_cards c
            JOIN speakers s    ON c.speaker_id  = s.speaker_id
            JOIN card_sources cs ON cs.card_id = c.card_id AND cs.source_order = 0
            JOIN results r     ON r.result_id   = cs.result_id
            WHERE c.suppressed = 0
            ORDER BY c.play_start, c.card_id
            """;
        using var r = cmd.ExecuteReader();
        while (r.Read())
            rows.Add((r.GetString(0), r.GetDouble(1), r.GetDouble(2),
                      r.IsDBNull(3) ? "" : r.GetString(3)));
        return rows;
    }

    /// <summary>Returns non-suppressed rows from the overlay for the summary grid.</summary>
    public List<IReadOnlyDictionary<string, string>> GetTranscript()
    {
        var rows = new List<IReadOnlyDictionary<string, string>>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT s.name, c.play_start, c.play_end,
                   coalesce(c.content, coalesce(r.asr_content, ''))
            FROM segment_cards c
            JOIN speakers s    ON c.speaker_id  = s.speaker_id
            JOIN card_sources cs ON cs.card_id = c.card_id AND cs.source_order = 0
            JOIN results r     ON r.result_id   = cs.result_id
            WHERE c.suppressed = 0
            ORDER BY c.play_start, c.card_id
            """;
        using var r = cmd.ExecuteReader();
        while (r.Read())
        {
            rows.Add(new Dictionary<string, string>
            {
                ["speaker_name"] = r.GetString(0),
                ["start_time"]   = FormatTimeF(r.GetDouble(1)),
                ["end_time"]     = FormatTimeF(r.GetDouble(2)),
                ["content"]      = r.IsDBNull(3) ? "" : r.GetString(3),
            });
        }
        return rows;
    }

    /// <summary>Returns ordered content strings for all result rows (used for display during transcription).</summary>
    public List<string> GetResultContents()
    {
        var result = new List<string>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT coalesce(content, '') FROM results ORDER BY result_id";
        using var r = cmd.ExecuteReader();
        while (r.Read())
            result.Add(r.GetString(0));
        return result;
    }

    public string? GetAudioFilePath() => GetMetadata("audio_file");

    // ── Check helpers ─────────────────────────────────────────────────────────

    /// <summary>Returns true if the SHA-256 in metadata matches the file.</summary>
    public bool CheckMetadata(string audioPath)
    {
        string? stored = GetMetadata("audio_file_sha256sum");
        if (stored is null) return false;
        return stored.Equals(AudioUtils.Sha256Checksum(audioPath),
            StringComparison.OrdinalIgnoreCase);
    }

    public bool CheckDiarization() =>
        GetMetadata("diarization_complete") == "1";

    public void MarkDiarizationComplete()
    {
        if (GetMetadata("diarization_complete") is null)
            InsertMetadata("diarization_complete", "1");
        else
            UpdateMetadata("diarization_complete", "1");
    }

    public bool CheckSpeakers()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT count(*) FROM speakers";
        return (long)(cmd.ExecuteScalar() ?? 0L) > 0;
    }

    /// <summary>
    /// Returns (allDone, startAtIndex).
    /// Mirrors sql.py check_asr().
    /// </summary>
    public (bool done, int startAt) CheckAsr()
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            SELECT
                (SELECT count(*) FROM results)                          AS results_count,
                (SELECT count(*) FROM results WHERE tokens IS NOT NULL) AS filled_results_count
            """;
        using var r = cmd.ExecuteReader();
        if (!r.Read()) return (false, 0);
        long total  = r.GetInt64(0);
        long filled = r.GetInt64(1);
        if (total == 0)      return (false, 0);
        if (filled < total)  return (false, (int)filled);
        return (true, (int)filled);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void Execute(string sql)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    private static string FormatTimeF(double seconds)
    {
        int    h   = (int)(seconds / 3600);
        int    m   = (int)(seconds % 3600 / 60);
        double sec = seconds % 60;
        return h > 0 ? $"{h}:{m:D2}:{sec:00.00}" : $"{m:D2}:{sec:00.00}";
    }

    private static IReadOnlyList<T> ParseJsonList<T>(string? json)
    {
        if (string.IsNullOrEmpty(json)) return [];
        try { return JsonSerializer.Deserialize<List<T>>(json) ?? []; }
        catch { return []; }
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose() => _conn.Dispose();
}

// ── Data records shared between DB and editor ─────────────────────────────────

/// <summary>Describes one source result that contributes tokens to a card.</summary>
internal record CardSource(
    int  ResultId,
    int  TokenStart,
    int? TokenEnd,
    int  TsFrameOffset,
    int  SourceOrder);

/// <summary>Aggregated card data returned by <see cref="TranscriptionDb.GetCardsForEditor"/>.</summary>
internal record CardQueryRow(
    int     CardId,
    double  SortOrder,
    int     SpeakerId,
    string  SpeakerName,
    double  PlayStart,
    double  PlayEnd,
    string  Content,     // display content (card override if set, else decoded from ASR)
    string? RawContent,  // raw DB value: null if not user-edited
    bool    Verified,
    bool    IsSuppressed,
    IReadOnlyList<CardSource> Sources,
    IReadOnlyList<int>        Tokens,
    IReadOnlyList<int>        Timestamps,
    IReadOnlyList<float>      Logprobs,
    string  AsrContent);
