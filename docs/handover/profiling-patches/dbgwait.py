f='ggml/src/ggml-rknpu2/ggml-rknpu2.cpp'; s=open(f).read()
# bucket wait time and node time by node-name prefix
s=s.replace("""static std::atomic<uint64_t> g_rknpu_wait_ns{0};""","""static std::atomic<uint64_t> g_rknpu_wait_ns{0};
#include <map>
static std::map<std::string, std::pair<double,double>> g_dbg_wait;
static void dbg_dump() { std::vector<std::pair<std::string, std::pair<double,double>>> v(g_dbg_wait.begin(), g_dbg_wait.end()); std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.second > b.second.second; }); for (auto& kv : v) fprintf(stderr, "DBGW %-28s wait %8.1f ms  node %8.1f ms\\n", kv.first.c_str(), kv.second.first / 1e6, kv.second.second / 1e6); }""",1)
s=s.replace("""        if (node->op != GGML_OP_MUL_MAT) continue;""","""        const uint64_t dbg_w0 = g_rknpu_wait_ns.load(); const auto dbg_t0 = std::chrono::steady_clock::now();
        struct DbgGuard { const ggml_tensor* nd; uint64_t w0; std::chrono::steady_clock::time_point t0; ~DbgGuard() { std::string k(nd->name); auto p = k.find('-'); if (p != std::string::npos) k = k.substr(0, p); if (k.rfind("node_", 0) == 0) k = std::string("op:") + ggml_op_name(nd->op) + (nd->op == GGML_OP_MUL_MAT ? std::string(" K=") + std::to_string(nd->src[0]->ne[0]) + " N=" + std::to_string(nd->src[0]->ne[1]) : std::string()); auto& e = g_dbg_wait[k]; e.first += (double)(g_rknpu_wait_ns.load() - w0); e.second += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); } } dbg_guard{node, dbg_w0, dbg_t0};
        if (node->op != GGML_OP_MUL_MAT) continue;""",1)
s=s.replace("""    g_rknpu_profile.maybe_print();""","""    g_rknpu_profile.maybe_print(); { static auto last = std::chrono::steady_clock::now(); if (std::chrono::steady_clock::now() - last > std::chrono::seconds(10)) { last = std::chrono::steady_clock::now(); dbg_dump(); } }""",1)
open(f,'w').write(s)
