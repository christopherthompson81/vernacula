using Microsoft.Data.Sqlite;
using Vernacula.App.Models;
using System.Globalization;

namespace Vernacula.App.Services;

/// <summary>
/// Application-level job history database.
/// Port of Electron's controlDatabase.ts ControlDB class.
/// </summary>
internal sealed class ControlDb : IDisposable
{
    private readonly SqliteConnection _conn;

    public ControlDb(string dbPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(dbPath)!);
        _conn = new SqliteConnection($"Data Source={dbPath}");
        _conn.Open();
        Execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title                   TEXT NOT NULL,
                results_file                TEXT NOT NULL,
                transcription_run_datestamp TEXT,
                audio_file_path             TEXT NOT NULL,
                audio_file_sha256sum        TEXT NOT NULL,
                audio_file_datestamp        TEXT,
                status                      TEXT NOT NULL DEFAULT 'pending',
                created_at                  TEXT NOT NULL
            )
            """);

        // Migrate: add stream_index if it does not exist yet
        try { Execute("ALTER TABLE jobs ADD COLUMN stream_index INTEGER NOT NULL DEFAULT -1"); }
        catch (SqliteException) { /* column already present — ignore */ }

        // Migrate: add error_message if it does not exist yet
        try { Execute("ALTER TABLE jobs ADD COLUMN error_message TEXT"); }
        catch (SqliteException) { /* column already present — ignore */ }

        // Migrate: add run_time_seconds if it does not exist yet
        try { Execute("ALTER TABLE jobs ADD COLUMN run_time_seconds INTEGER"); }
        catch (SqliteException) { /* column already present — ignore */ }

        // Migrate: add per-job ASR language snapshot if it does not exist yet
        try { Execute("ALTER TABLE jobs ADD COLUMN asr_language_code TEXT NOT NULL DEFAULT 'auto'"); }
        catch (SqliteException) { /* column already present — ignore */ }

        // Migrate: add per-job ASR model snapshot if it does not exist yet
        try { Execute("ALTER TABLE jobs ADD COLUMN asr_model_name TEXT NOT NULL DEFAULT 'nvidia/parakeet-tdt-0.6b-v3'"); }
        catch (SqliteException) { /* column already present — ignore */ }

        // Migrate: job kind + per-job TTS settings + output duration. Every row that predates
        // this column is an ASR job, hence the default. audio_file_path keeps its name and
        // holds the input document for TTS jobs (see JobRecord.AudioFilePath).
        foreach (string ddl in new[]
        {
            "ALTER TABLE jobs ADD COLUMN job_kind TEXT NOT NULL DEFAULT 'asr'",
            "ALTER TABLE jobs ADD COLUMN tts_backend TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE jobs ADD COLUMN tts_language TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE jobs ADD COLUMN tts_voice TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE jobs ADD COLUMN tts_speed REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE jobs ADD COLUMN tts_num_step INTEGER NOT NULL DEFAULT 32",
            "ALTER TABLE jobs ADD COLUMN output_duration_seconds REAL",
            // Per-kind engine settings as one JSON blob, superseding the five tts_* columns
            // above (issue #130). Those stay in the schema, unwritten, so a database written
            // by this version still opens in an older build; nothing reads them any more
            // except the backfill below.
            "ALTER TABLE jobs ADD COLUMN job_settings TEXT",
        })
        {
            try { Execute(ddl); }
            catch (SqliteException) { /* column already present — ignore */ }
        }

        BackfillTtsJobSettings();
    }

    /// <summary>
    /// Moves TTS rows written before <c>job_settings</c> existed into it. Idempotent — it only
    /// touches rows that have no JSON yet — so it is safe to run on every open, which also
    /// covers a process that died between the ALTER and the copy.
    /// </summary>
    private void BackfillTtsJobSettings()
    {
        var legacy = new List<(int JobId, TtsJobSettings Tts)>();
        using (var read = _conn.CreateCommand())
        {
            read.CommandText = """
                SELECT job_id, tts_backend, tts_language, tts_voice, tts_speed, tts_num_step
                FROM jobs
                WHERE job_kind = 'tts' AND job_settings IS NULL
                """;
            using var r = read.ExecuteReader();
            while (r.Read())
                legacy.Add((r.GetInt32(0), new TtsJobSettings(
                    r.IsDBNull(1) ? "" : r.GetString(1),
                    r.IsDBNull(2) ? "" : r.GetString(2),
                    r.IsDBNull(3) ? "" : r.GetString(3),
                    r.IsDBNull(4) ? 1.0f : (float)r.GetDouble(4),
                    r.IsDBNull(5) ? 32 : r.GetInt32(5))));
        }
        if (legacy.Count == 0) return;

        using var tx = _conn.BeginTransaction();
        foreach (var (jobId, tts) in legacy)
        {
            using var upd = _conn.CreateCommand();
            upd.Transaction = tx;
            upd.CommandText = "UPDATE jobs SET job_settings = $js WHERE job_id = $id";
            upd.Parameters.AddWithValue("$js", SerializeTts(tts));
            upd.Parameters.AddWithValue("$id", jobId);
            upd.ExecuteNonQuery();
        }
        tx.Commit();
    }

    // The jobs table stores per-kind engine settings as JSON so that a new engine knob is one
    // edit to the record rather than a column, two SQL statements, a parameter, a property and
    // a reader ordinal. System.Text.Json is culture-invariant, which the column needs to be.
    private static string SerializeTts(TtsJobSettings tts) =>
        System.Text.Json.JsonSerializer.Serialize(tts);

    private static TtsJobSettings? DeserializeTts(string? json)
    {
        if (string.IsNullOrWhiteSpace(json)) return null;
        try { return System.Text.Json.JsonSerializer.Deserialize<TtsJobSettings>(json); }
        catch (System.Text.Json.JsonException) { return null; }   // unreadable row → treat as unset
    }

    /// <summary>
    /// Inserts a new job or updates the status of an existing one (matched by results_file).
    /// Returns the job_id.
    /// </summary>
    public int UpsertJob(string title, string resultsFile, string audioPath,
                         string sha256, string audioDateStamp, string runDateStamp,
                         string asrLanguageCode = "auto",
                         string asrModelName = "nvidia/parakeet-tdt-0.6b-v3")
    {
        using var check = _conn.CreateCommand();
        check.CommandText = "SELECT job_id FROM jobs WHERE results_file = $rf";
        check.Parameters.AddWithValue("$rf", resultsFile);
        var existing = check.ExecuteScalar();

        if (existing is long existingId)
        {
            using var upd = _conn.CreateCommand();
            upd.CommandText = """
                UPDATE jobs
                SET status = 'running',
                    transcription_run_datestamp = $ts,
                    job_title = $jt,
                    asr_language_code = $lc,
                    asr_model_name = $am
                WHERE job_id = $id
                """;
            upd.Parameters.AddWithValue("$ts", runDateStamp);
            upd.Parameters.AddWithValue("$jt", title);
            upd.Parameters.AddWithValue("$lc", asrLanguageCode);
            upd.Parameters.AddWithValue("$am", asrModelName);
            upd.Parameters.AddWithValue("$id", existingId);
            upd.ExecuteNonQuery();
            return (int)existingId;
        }

        using var ins = _conn.CreateCommand();
        ins.CommandText = """
            INSERT INTO jobs
                (job_title, results_file, transcription_run_datestamp,
                 audio_file_path, audio_file_sha256sum, audio_file_datestamp,
                 status, created_at, asr_language_code, asr_model_name)
            VALUES ($jt, $rf, $ts, $ap, $sh, $ad, 'running', $ca, $lc, $am)
            """;
        ins.Parameters.AddWithValue("$jt", title);
        ins.Parameters.AddWithValue("$rf", resultsFile);
        ins.Parameters.AddWithValue("$ts", runDateStamp);
        ins.Parameters.AddWithValue("$ap", audioPath);
        ins.Parameters.AddWithValue("$sh", sha256);
        ins.Parameters.AddWithValue("$ad", audioDateStamp);
        ins.Parameters.AddWithValue("$ca", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
        ins.Parameters.AddWithValue("$lc", asrLanguageCode);
        ins.Parameters.AddWithValue("$am", asrModelName);
        ins.ExecuteNonQuery();

        using var lastId = _conn.CreateCommand();
        lastId.CommandText = "SELECT last_insert_rowid()";
        return (int)(long)lastId.ExecuteScalar()!;
    }

    /// <summary>
    /// Inserts a brand-new job with 'queued' status. If a job for the same
    /// results_file already exists, returns its existing job_id unchanged.
    /// </summary>
    public int InsertNewJob(string title, string resultsFile, string audioPath,
                            string sha256, string audioDateStamp, int streamIndex = -1,
                            string asrLanguageCode = "auto",
                            string asrModelName = "nvidia/parakeet-tdt-0.6b-v3")
        => InsertJob(JobKind.Asr, title, resultsFile, audioPath, sha256, audioDateStamp,
                     streamIndex, asrLanguageCode, asrModelName, settingsJson: null);

    /// <summary>
    /// Inserts a brand-new text-to-speech job with 'queued' status. A job for the same
    /// results_file (same document + same backend/voice/language, see
    /// JobQueueService.TtsResultsFileName) is updated in place and its job_id returned, so
    /// re-adding a document re-renders rather than duplicating the row.
    /// </summary>
    public int InsertNewTtsJob(string title, string resultsFile, string documentPath,
                               string sha256, string documentDateStamp, TtsJobSettings tts)
        => InsertJob(JobKind.Tts, title, resultsFile, documentPath, sha256, documentDateStamp,
                     streamIndex: -1, asrLanguageCode: "auto",
                     asrModelName: "nvidia/parakeet-tdt-0.6b-v3",
                     settingsJson: SerializeTts(tts));

    /// <summary>
    /// The one upsert-by-results_file behind both. Re-adding a job re-runs it: the row goes
    /// back to 'queued' with its previous outcome cleared, or Home would keep showing the old
    /// result (and its Resume button) while the new run overwrote the files underneath it.
    /// </summary>
    private int InsertJob(JobKind kind, string title, string resultsFile, string inputPath,
                          string sha256, string inputDateStamp, int streamIndex,
                          string asrLanguageCode, string asrModelName, string? settingsJson)
    {
        string kindText = kind == JobKind.Tts ? "tts" : "asr";
        object settingsValue = (object?)settingsJson ?? DBNull.Value;

        using var check = _conn.CreateCommand();
        check.CommandText = "SELECT job_id FROM jobs WHERE results_file = $rf";
        check.Parameters.AddWithValue("$rf", resultsFile);
        if (check.ExecuteScalar() is long existingId)
        {
            using var upd = _conn.CreateCommand();
            upd.CommandText = """
                UPDATE jobs
                SET job_title = $jt,
                    audio_file_path = $ap,
                    audio_file_sha256sum = $sh,
                    audio_file_datestamp = $ad,
                    stream_index = $si,
                    asr_language_code = $lc,
                    asr_model_name = $am,
                    job_kind = $jk,
                    job_settings = $js,
                    output_duration_seconds = NULL,
                    status = 'queued',
                    error_message = NULL,
                    run_time_seconds = NULL,
                    transcription_run_datestamp = NULL
                WHERE job_id = $id
                """;
            AddCommonParameters(upd, title, inputPath, sha256, inputDateStamp,
                                streamIndex, asrLanguageCode, asrModelName, kindText, settingsValue);
            upd.Parameters.AddWithValue("$id", existingId);
            upd.ExecuteNonQuery();
            return (int)existingId;
        }

        using var ins = _conn.CreateCommand();
        ins.CommandText = """
            INSERT INTO jobs
                (job_title, results_file, transcription_run_datestamp,
                 audio_file_path, audio_file_sha256sum, audio_file_datestamp,
                 status, created_at, stream_index, asr_language_code, asr_model_name,
                 job_kind, job_settings)
            VALUES ($jt, $rf, NULL, $ap, $sh, $ad, 'queued', $ca, $si, $lc, $am, $jk, $js)
            """;
        AddCommonParameters(ins, title, inputPath, sha256, inputDateStamp,
                            streamIndex, asrLanguageCode, asrModelName, kindText, settingsValue);
        ins.Parameters.AddWithValue("$rf", resultsFile);
        ins.Parameters.AddWithValue("$ca", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
        ins.ExecuteNonQuery();

        using var lastId = _conn.CreateCommand();
        lastId.CommandText = "SELECT last_insert_rowid()";
        return (int)(long)lastId.ExecuteScalar()!;
    }

    private static void AddCommonParameters(
        SqliteCommand cmd, string title, string inputPath, string sha256, string inputDateStamp,
        int streamIndex, string asrLanguageCode, string asrModelName, string kindText,
        object settingsValue)
    {
        cmd.Parameters.AddWithValue("$jt", title);
        cmd.Parameters.AddWithValue("$ap", inputPath);
        cmd.Parameters.AddWithValue("$sh", sha256);
        cmd.Parameters.AddWithValue("$ad", inputDateStamp);
        cmd.Parameters.AddWithValue("$si", streamIndex);
        cmd.Parameters.AddWithValue("$lc", asrLanguageCode);
        cmd.Parameters.AddWithValue("$am", asrModelName);
        cmd.Parameters.AddWithValue("$jk", kindText);
        cmd.Parameters.AddWithValue("$js", settingsValue);
    }

    /// <summary>TTS: the rendered audio length recorded for a job, if any.</summary>
    public double? GetJobOutputDuration(int jobId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT output_duration_seconds FROM jobs WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$id", jobId);
        var result = cmd.ExecuteScalar();
        return result is DBNull or null ? null : Convert.ToDouble(result);
    }

    /// <summary>TTS: records how long the rendered audio is once a job completes.</summary>
    public void UpdateJobOutputDuration(int jobId, double seconds)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE jobs SET output_duration_seconds = $d WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$d",  seconds);
        cmd.Parameters.AddWithValue("$id", jobId);
        cmd.ExecuteNonQuery();
    }

    /// <summary>Sets a job to 'running' and records when the run started.</summary>
    public void SetJobRunning(int jobId, string runStamp)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            UPDATE jobs
            SET status = 'running', transcription_run_datestamp = $ts
            WHERE job_id = $id
            """;
        cmd.Parameters.AddWithValue("$ts", runStamp);
        cmd.Parameters.AddWithValue("$id", jobId);
        cmd.ExecuteNonQuery();
    }

    public void UpdateJobTitle(int jobId, string title)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "UPDATE jobs SET job_title = $t WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$t",  title);
        cmd.Parameters.AddWithValue("$id", jobId);
        cmd.ExecuteNonQuery();
    }

    /// <summary>
    /// Updates a job's ASR backend and forced-language so a subsequent
    /// requeue runs with the new configuration. Intended for the Results
    /// view "Reprocess with <backend>" remedy after LID detected a language
    /// the original backend couldn't handle.
    /// </summary>
    public void UpdateJobAsr(int jobId, string asrModelName, string asrLanguageCode)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            UPDATE jobs
            SET asr_model_name = $m, asr_language_code = $lc
            WHERE job_id = $id
            """;
        cmd.Parameters.AddWithValue("$m",  asrModelName);
        cmd.Parameters.AddWithValue("$lc", asrLanguageCode);
        cmd.Parameters.AddWithValue("$id", jobId);
        cmd.ExecuteNonQuery();
    }

    public void UpdateJobStatus(int jobId, JobStatus status, string? errorMessage = null,
                                int? runTimeSeconds = null)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = """
            UPDATE jobs
            SET status = $s, error_message = $e, run_time_seconds = $rt
            WHERE job_id = $id
            """;
        cmd.Parameters.AddWithValue("$s",  status.ToString().ToLowerInvariant());
        cmd.Parameters.AddWithValue("$e",  (object?)errorMessage   ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$rt", (object?)runTimeSeconds ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$id", jobId);
        cmd.ExecuteNonQuery();
    }

    public string? GetJobError(int jobId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT error_message FROM jobs WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$id", jobId);
        var result = cmd.ExecuteScalar();
        return result is DBNull or null ? null : (string)result;
    }

    public string GetJobAsrLanguageCode(int jobId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT asr_language_code FROM jobs WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$id", jobId);
        var result = cmd.ExecuteScalar();
        return result is DBNull or null or "" ? "auto" : (string)result;
    }

    public string GetJobAsrModelName(int jobId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT asr_model_name FROM jobs WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$id", jobId);
        var result = cmd.ExecuteScalar();
        return result is DBNull or null or "" ? "nvidia/parakeet-tdt-0.6b-v3" : (string)result;
    }

    public DateTime? GetJobRunStartedAt(int jobId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT transcription_run_datestamp FROM jobs WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$id", jobId);
        var result = cmd.ExecuteScalar();
        if (result is DBNull or null)
            return null;

        string value = (string)result;
        return DateTime.TryParseExact(
            value,
            "yyyy-MM-dd HH:mm:ss",
            CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeLocal,
            out DateTime parsed)
            ? parsed
            : null;
    }

    public List<JobRecord> GetJobs()
    {
        var jobs = new List<JobRecord>();
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "SELECT * FROM jobs ORDER BY created_at DESC";
        using var r = cmd.ExecuteReader();
        while (r.Read())
            jobs.Add(ReadRow(r));
        return jobs;
    }

    public void DeleteJob(int jobId)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = "DELETE FROM jobs WHERE job_id = $id";
        cmd.Parameters.AddWithValue("$id", jobId);
        cmd.ExecuteNonQuery();
    }

    private static JobRecord ReadRow(SqliteDataReader r)
    {
        string? GetNullable(string col) =>
            r.IsDBNull(r.GetOrdinal(col)) ? null : r.GetString(r.GetOrdinal(col));

        static DateTime? ParseStamp(string? value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return null;

            return DateTime.TryParseExact(
                value,
                "yyyy-MM-dd HH:mm:ss",
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeLocal,
                out DateTime parsed)
                ? parsed
                : null;
        }

        int siOrd = r.GetOrdinal("stream_index");
        int odOrd = r.GetOrdinal("output_duration_seconds");
        string? runStamp = GetNullable("transcription_run_datestamp");
        return new JobRecord
        {
            JobId                     = r.GetInt32(r.GetOrdinal("job_id")),
            JobTitle                  = r.GetString(r.GetOrdinal("job_title")),
            Kind                      = string.Equals(GetNullable("job_kind"), "tts", StringComparison.OrdinalIgnoreCase)
                                            ? JobKind.Tts : JobKind.Asr,
            TtsSettings               = DeserializeTts(GetNullable("job_settings")),
            OutputDurationSeconds     = r.IsDBNull(odOrd) ? null : r.GetDouble(odOrd),
            ResultsFile               = r.GetString(r.GetOrdinal("results_file")),
            AudioFilePath             = r.GetString(r.GetOrdinal("audio_file_path")),
            AudioFileSha256Sum        = r.GetString(r.GetOrdinal("audio_file_sha256sum")),
            AsrModelName              = GetNullable("asr_model_name") ?? "nvidia/parakeet-tdt-0.6b-v3",
            AsrLanguageCode           = GetNullable("asr_language_code") ?? "auto",
            AudioFileDatestamp        = GetNullable("audio_file_datestamp"),
            TranscriptionRunDatestamp = runStamp,
            TranscriptionRunStartedAt = ParseStamp(runStamp),
            Status = Enum.Parse<JobStatus>(
                r.GetString(r.GetOrdinal("status")), ignoreCase: true),
            CreatedAt         = r.GetString(r.GetOrdinal("created_at")),
            AudioStreamIndex  = r.IsDBNull(siOrd) ? -1 : r.GetInt32(siOrd),
            ErrorMessage      = GetNullable("error_message"),
            RunTimeSeconds    = r.IsDBNull(r.GetOrdinal("run_time_seconds"))
                                    ? null
                                    : r.GetInt32(r.GetOrdinal("run_time_seconds")),
        };
    }

    private void Execute(string sql)
    {
        using var cmd = _conn.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    public void Dispose() => _conn.Dispose();
}
