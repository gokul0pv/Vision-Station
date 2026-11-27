#include "results.hpp"

#include <sqlite3.h>
#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

namespace vc {

static const char* SCHEMA = R"(
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_uid     TEXT,
    ts          REAL    NOT NULL,
    iso         TEXT    NOT NULL,
    project     TEXT    NOT NULL,
    verdict     TEXT    NOT NULL,
    total_ms    REAL,
    source      TEXT,
    image_path  TEXT,
    operator    TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ts      ON runs(ts);
CREATE INDEX IF NOT EXISTS idx_runs_verdict ON runs(verdict);
CREATE TABLE IF NOT EXISTS tool_results (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    tool_id  TEXT    NOT NULL,
    type     TEXT,
    status   TEXT,
    score    REAL,
    ms       REAL,
    message  TEXT,
    values_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_tr_run    ON tool_results(run_id);
CREATE INDEX IF NOT EXISTS idx_tr_tool   ON tool_results(tool_id, status);
)";

struct Item {
    Json result;
    cv::Mat picture;
    std::string op;
};

struct ResultStore::Impl {
    std::string folder, project, archive;
    int max_archived;
    fs::path dir, images;
    std::mutex mx;
    std::condition_variable cv;
    std::deque<Item> queue;
    bool stop = false;
    std::thread th;
    std::string error;
    std::atomic<long long> written{0};
    int since_prune = 0;

    void run() {
        sqlite3* db = nullptr;
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (sqlite3_open_v2((dir / "results.db").u8string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK) {
            set_error("cannot open results.db");
            if (db) sqlite3_close(db);
            db = nullptr;
        } else {
            sqlite3_busy_timeout(db, 5000);
            sqlite3_exec(db, "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;", nullptr, nullptr, nullptr);
            char* e = nullptr;
            if (sqlite3_exec(db, SCHEMA, nullptr, nullptr, &e) != SQLITE_OK) {
                set_error(e ? e : "schema");
                sqlite3_free(e);
            }
        }
        for (;;) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(mx);
                cv.wait(lk, [&] { return stop || !queue.empty(); });
                if (queue.empty() && stop) break;
                it = std::move(queue.front());
                queue.pop_front();
            }
            if (!db) continue;
            try {
                write_one(db, it);
                written++;
            } catch (const std::exception& e) {
                set_error(std::string("write failed: ") + e.what());   // logging must never kill the engine
            }
        }
        if (db) sqlite3_close(db);
    }

    void set_error(const std::string& e) {
        std::lock_guard<std::mutex> lk(mx);
        error = e;
    }

    void write_one(sqlite3* db, const Item& it) {
        using namespace std::chrono;
        auto now = system_clock::now();
        double ts = duration<double>(now.time_since_epoch()).count();
        std::time_t tt = system_clock::to_time_t(now);
        std::tm lt = *std::localtime(&tt);
        char iso[32], day[16], hms[16];
        std::strftime(iso, sizeof iso, "%Y-%m-%dT%H:%M:%S", &lt);
        std::strftime(day, sizeof day, "%Y-%m-%d", &lt);
        std::strftime(hms, sizeof hms, "%H%M%S", &lt);
        int ms = (int)(duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000);
        const Json& r = it.result;
        std::string verdict = r.value("verdict", "");
        std::string img_path;
        bool keep = !it.picture.empty() && (archive == "all" || (archive == "fail" && verdict != "PASS"));
        if (keep) {
            fs::path d = images / day;
            std::error_code ec;
            fs::create_directories(d, ec);
            char name[96];
            std::snprintf(name, sizeof name, "%s_%03d_%s_%s.jpg", hms, ms, verdict.c_str(), r.value("id", "").substr(0, 8).c_str());
            std::vector<uint8_t> buf;
            if (cv::imencode(".jpg", it.picture, buf, {cv::IMWRITE_JPEG_QUALITY, 85})) {
                std::ofstream f(d / name, std::ios::binary);
                f.write((const char*)buf.data(), buf.size());
                if (f) {
                    img_path = fs::relative(d / name, fs::u8path(folder), ec).u8string();
                    since_prune++;
                }
            }
        }
        sqlite3_exec(db, "BEGIN", nullptr, nullptr, nullptr);
        sqlite3_stmt* st = nullptr;
        sqlite3_prepare_v2(db, "INSERT INTO runs (run_uid, ts, iso, project, verdict, total_ms, source, image_path, operator) VALUES (?,?,?,?,?,?,?,?,?)", -1, &st, nullptr);
        std::string uid = r.value("id", ""), src = r.value("source", "");
        sqlite3_bind_text(st, 1, uid.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(st, 2, ts);
        sqlite3_bind_text(st, 3, iso, -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 4, project.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 5, verdict.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(st, 6, r.value("total_ms", 0.0));
        sqlite3_bind_text(st, 7, src.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 8, img_path.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(st, 9, it.op.c_str(), -1, SQLITE_TRANSIENT);
        int rc = sqlite3_step(st);
        sqlite3_finalize(st);
        if (rc != SQLITE_DONE) {
            sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr);
            throw std::runtime_error(sqlite3_errmsg(db));
        }
        long long rid = sqlite3_last_insert_rowid(db);
        sqlite3_prepare_v2(db, "INSERT INTO tool_results (run_id, tool_id, type, status, score, ms, message, values_json) VALUES (?,?,?,?,?,?,?,?)", -1, &st, nullptr);
        for (auto& t : r["tools"]) {
            std::string a = t.value("tool_id", ""), b = t.value("type", ""), c = t.value("status", ""), m = t.value("message", "");
            std::string v = t.contains("values") ? t["values"].dump(-1, ' ', false, Json::error_handler_t::replace) : "{}";
            sqlite3_reset(st);
            sqlite3_bind_int64(st, 1, rid);
            sqlite3_bind_text(st, 2, a.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 3, b.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 4, c.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_double(st, 5, t.value("score", 0.0));
            sqlite3_bind_double(st, 6, t.value("ms", 0.0));
            sqlite3_bind_text(st, 7, m.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(st, 8, v.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_step(st);
        }
        sqlite3_finalize(st);
        sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
        if (since_prune >= 25) prune();
    }

    // the oldest archived pictures go once there are more than max_archived
    void prune() {
        since_prune = 0;
        std::vector<std::pair<fs::file_time_type, fs::path>> files;
        std::error_code ec;
        for (auto it = fs::recursive_directory_iterator(images, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            auto ext = it->path().extension().u8string();
            if (ext == ".jpg" || ext == ".png") files.emplace_back(fs::last_write_time(it->path(), ec), it->path());
        }
        if ((int)files.size() <= max_archived) return;
        std::sort(files.begin(), files.end());
        for (size_t i = 0; i < files.size() - max_archived; i++) fs::remove(files[i].second, ec);
    }
};

ResultStore::ResultStore(const std::string& side_folder, const std::string& project_id, const std::string& archive, int max_archived)
    : d_(std::make_unique<Impl>()) {
    d_->folder = side_folder;
    d_->project = project_id;
    d_->archive = archive;
    d_->max_archived = max_archived;
    d_->dir = fs::u8path(side_folder) / "results";
    d_->images = d_->dir / "images";
    d_->th = std::thread([this] { d_->run(); });
}

ResultStore::~ResultStore() {
    {
        std::lock_guard<std::mutex> lk(d_->mx);
        d_->stop = true;
    }
    d_->cv.notify_all();
    if (d_->th.joinable()) d_->th.join();
}

void ResultStore::write(const Json& inspection, const cv::Mat& picture, const std::string& op) {
    {
        std::lock_guard<std::mutex> lk(d_->mx);
        if (d_->queue.size() >= 512) d_->queue.pop_front();   // full: drop the oldest, never block the line
        d_->queue.push_back({inspection, picture, op});
    }
    d_->cv.notify_one();
}

std::string ResultStore::last_error() const {
    std::lock_guard<std::mutex> lk(d_->mx);
    return d_->error;
}

long long ResultStore::written() const { return d_->written.load(); }

}  // namespace vc
