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
    {
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
                    asr_model_name = $am
                WHERE job_id = $id
                """;
            upd.Parameters.AddWithValue("$jt", title);
            upd.Parameters.AddWithValue("$ap", audioPath);
            upd.Parameters.AddWithValue("$sh", sha256);
            upd.Parameters.AddWithValue("$ad", audioDateStamp);
            upd.Parameters.AddWithValue("$si", streamIndex);
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
                 status, created_at, stream_index, asr_language_code, asr_model_name)
            VALUES ($jt, $rf, NULL, $ap, $sh, $ad, 'queued', $ca, $si, $lc, $am)
            """;
        ins.Parameters.AddWithValue("$jt", title);
        ins.Parameters.AddWithValue("$rf", resultsFile);
        ins.Parameters.AddWithValue("$ap", audioPath);
        ins.Parameters.AddWithValue("$sh", sha256);
        ins.Parameters.AddWithValue("$ad", audioDateStamp);
        ins.Parameters.AddWithValue("$ca", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
        ins.Parameters.AddWithValue("$si", streamIndex);
        ins.Parameters.AddWithValue("$lc", asrLanguageCode);
        ins.Parameters.AddWithValue("$am", asrModelName);
        ins.ExecuteNonQuery();

        using var lastId = _conn.CreateCommand();
        lastId.CommandText = "SELECT last_insert_rowid()";
        return (int)(long)lastId.ExecuteScalar()!;
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
        string? runStamp = GetNullable("transcription_run_datestamp");
        return new JobRecord
        {
            JobId                     = r.GetInt32(r.GetOrdinal("job_id")),
            JobTitle                  = r.GetString(r.GetOrdinal("job_title")),
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
