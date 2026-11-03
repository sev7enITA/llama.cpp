#include "core.h"
#include "llama.h"
#include "nlohmann/json.hpp"
extern "C" {
#include "sha256.h"
}
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

using json = nlohmann::json;
using tokens = std::vector<llama_token>;
using clock_type = std::chrono::steady_clock;
static void require(bool ok, const char * msg) { if (!ok) throw std::runtime_error(msg); }
static double elapsed(clock_type::time_point t) { return std::chrono::duration<double,std::milli>(clock_type::now()-t).count(); }
static std::once_flag backend_once;

struct Engine {
    std::mutex mutex;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> closing{false};
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx{nullptr, llama_free};
    const llama_vocab * vocab = nullptr;
    tokens cached;
    // This engine is greedy-only. Retain the argmax, not a full vocabulary
    // buffer that would be copied and scanned again before every token.
    llama_token next = -1;
    std::string fingerprint;
    int nctx = 0;
    double sync_ms = 0, decode_ms = 0;
    uint64_t decode_calls = 0, decode_tokens = 0, sync_tokens = 0;
    bool syncing = false;

    static bool abort(void * p) { return static_cast<Engine *>(p)->cancelled.load(); }
    static bool load_progress(float, void * p) { return !abort(p); }
    void check() { require(!cancelled.load(), "cancelled"); }
    void valid(const tokens & ids) {
        for (auto id : ids) require(id >= 0 && id < llama_vocab_n_tokens(vocab), "token outside vocabulary");
    }
    llama_token best(const float * logits) {
        require(logits != nullptr, "missing target logits");
        auto count = llama_vocab_n_tokens(vocab);
        llama_token selected = 0;
        for (int i=0;i<count;++i) {
            require(std::isfinite(logits[i]), "non-finite logits");
            if (logits[i] > logits[selected]) selected=i;
        }
        return selected;
    }
    std::string vocab_hash() {
        // Canonical token-ID semantics. Host alone tokenizes/formats prompts.
        sha256_t hash; sha256_init(&hash);
        auto add = [&](const std::string & s) {
            auto n=std::to_string(s.size())+":";
            sha256_update(&hash,reinterpret_cast<const unsigned char *>(n.data()),n.size());
            sha256_update(&hash,reinterpret_cast<const unsigned char *>(s.data()),s.size());
        };
        add(std::to_string(llama_vocab_type(vocab)));
        add(std::to_string(llama_vocab_bos(vocab))); add(std::to_string(llama_vocab_eos(vocab)));
        add(std::to_string(llama_vocab_get_add_bos(vocab))); add(std::to_string(llama_vocab_get_add_eos(vocab)));
        for (int i=0;i<llama_vocab_n_tokens(vocab);++i) {
            const char * text=llama_vocab_get_text(vocab,i);
            add(text ? text : ""); add(std::to_string(llama_vocab_get_attr(vocab,i)));
            add(std::to_string(llama_vocab_is_eog(vocab,i)));
        }
        unsigned char bytes[32]; sha256_final(&hash,bytes);
        const char * hex="0123456789abcdef"; std::string value;
        for (auto b:bytes) {value+=hex[b>>4];value+=hex[b&15];} return value;
    }
    json load(const json & req) {
        ctx.reset(); model.reset(); cached.clear();next=-1;vocab=nullptr;
        nctx=req.value("n_ctx",2048); int threads=req.value("threads",4),gpu=req.value("gpu_layers",0);
        require(nctx>=256 && nctx<=8192 && threads>=1 && threads<=16 && gpu>=0 && gpu<=999,"invalid model limits");
        std::call_once(backend_once,[]{llama_backend_init();});
        auto mp=llama_model_default_params();mp.n_gpu_layers=gpu;
        mp.progress_callback=load_progress;mp.progress_callback_user_data=this;
        model.reset(llama_model_load_from_file(req.at("path").get<std::string>().c_str(),mp));
        check();
        require(bool(model),"model could not be loaded");
        require(!llama_model_is_recurrent(model.get()) && !llama_model_is_hybrid(model.get()),"recurrent/hybrid models require checkpoint rollback and are not qualified");
        require(!llama_model_has_encoder(model.get()),"encoder-decoder models are not qualified");
        auto cp=llama_context_default_params();cp.n_ctx=nctx;cp.n_batch=512;cp.n_ubatch=512;
        cp.n_threads=threads;cp.n_threads_batch=threads;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.abort_callback=abort;cp.abort_callback_data=this;
        ctx.reset(llama_init_from_model(model.get(),cp));require(bool(ctx),"context allocation failed");
        check();nctx=int(llama_n_ctx(ctx.get()));
        vocab=llama_model_get_vocab(model.get());fingerprint=vocab_hash();
        return {{"tokenizer_fingerprint",fingerprint},{"vocab_size",llama_vocab_n_tokens(vocab)},
                {"n_ctx",llama_n_ctx(ctx.get())},{"gpu_layers_requested",gpu},{"runtime",llama_version()}};
    }
    void reset() { llama_memory_clear(llama_get_memory(ctx.get()),true);cached.clear();next=-1; }
    std::vector<llama_token> decode(const tokens & input, bool all) {
        check(); valid(input); require(!input.empty() && input.size()<=512,"invalid batch");
        require(cached.size()+input.size()<=size_t(nctx),"context limit reached");
        auto batch=llama_batch_init(int(input.size()),0,1);batch.n_tokens=int(input.size());
        for (int i=0;i<batch.n_tokens;++i) {
            batch.token[i]=input[i];batch.pos[i]=int(cached.size())+i;
            batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=all || i==batch.n_tokens-1;
        }
        next=-1;
        auto began=clock_type::now();
        int rc=llama_decode(ctx.get(),batch);llama_batch_free(batch);
        decode_ms+=elapsed(began);++decode_calls;decode_tokens+=input.size();
        if(syncing)sync_tokens+=input.size();
        if(rc!=0){reset();throw std::runtime_error(cancelled.load()?"cancelled":"llama_decode failed; cache reset");}
        cached.insert(cached.end(),input.begin(),input.end());
        std::vector<llama_token> choices;
        if(all) for(int i=0;i<int(input.size());++i) choices.push_back(best(llama_get_logits_ith(ctx.get(),i)));
        next=all ? choices.back() : best(llama_get_logits_ith(ctx.get(),-1));
        return choices;
    }
    void sync(const tokens & prefix) {
        struct SyncTimer {
            Engine & e; clock_type::time_point started=clock_type::now();
            explicit SyncTimer(Engine & value):e(value){e.syncing=true;}
            ~SyncTimer(){e.sync_ms+=elapsed(started);e.syncing=false;}
        } timer(*this);
        check(); valid(prefix);require(!prefix.empty() && prefix.size()<size_t(nctx),"prefix must be nonempty and leave context room");
        if(prefix==cached && next>=0)return;
        size_t keep=0;while(keep<prefix.size() && keep<cached.size() && prefix[keep]==cached[keep])++keep;
#ifdef NODELIFT_EXPERIMENTAL_DEFERRED_DRAFT
        // Only replay the anchor when no new suffix will provide fresh logits.
        if(keep>0 && keep==prefix.size())--keep;
#else
        // Re-evaluate the last retained token to recover the correct next-token logits.
        if(keep>0 && (keep<cached.size() || keep==prefix.size() || next<0))--keep;
#endif
        if(keep<cached.size() && !llama_memory_seq_rm(llama_get_memory(ctx.get()),0,int(keep),-1)) {reset();keep=0;}
        cached.resize(keep);next=-1;
        while(keep<prefix.size()) {
            size_t end=std::min(keep+size_t(512),prefix.size());
            decode(tokens(prefix.begin()+keep,prefix.begin()+end),false);keep=end;
        }
    }
    tokens draft(const tokens & prefix,int count) {
        require(count>=1 && count<=512,"invalid draft budget");sync(prefix);tokens output;
        for(int i=0;i<count && cached.size()<size_t(nctx);++i) {
            check();require(next>=0,"missing next token");auto id=next;output.push_back(id);
#ifdef NODELIFT_EXPERIMENTAL_DEFERRED_DRAFT
            // The final prediction is already known. Evaluate it only if a later
            // request commits it; rejected draft tails never enter the KV cache.
            if(llama_vocab_is_eog(vocab,id) || i+1==count)break;
            decode({id},false);
#else
            decode({id},false);
            if(llama_vocab_is_eog(vocab,id))break;
#endif
        } return output;
    }
    // Serial verification is deliberately opt-in: it prioritizes matching
    // the one-token decode path over batch throughput. It cannot accelerate
    // autoregressive generation and does not replace hardware qualification.
    json verify_serial(const tokens & prefix, const tokens & proposal, int remaining) {
        require(proposal.size()<=16 && remaining>=1 && remaining<=512,"invalid verification budget");
        valid(proposal);valid(prefix);
        require(!prefix.empty() && prefix.size()<size_t(nctx) && prefix.size()+proposal.size()<=size_t(nctx),"verification would exceed context");
        // A fully accepted proposal may add a bonus token. Never emit beyond
        // the context, even when the caller supplies a larger output budget.
        remaining=std::min(remaining,nctx-int(prefix.size()));
        sync(prefix);tokens output;int accepted=0;
        if(proposal.empty()) output.push_back(next);
        else for(size_t i=0;i<proposal.size();++i) {
            check();require(next>=0,"missing next token");auto expected=next;
            if(proposal[i]!=expected){output.push_back(expected);break;}
            output.push_back(expected);++accepted;
            if(llama_vocab_is_eog(vocab,expected) || int(output.size())>=remaining)break;
            decode({expected},false);
            if(i+1==proposal.size())output.push_back(next);
        }
        if(output.size()>size_t(remaining))output.resize(remaining);
        for(size_t i=0;i<output.size();++i)if(llama_vocab_is_eog(vocab,output[i])){output.resize(i+1);break;}
        accepted=std::min(accepted,int(output.size()));
        return {{"tokens",output},{"accepted",accepted},{"proposed",proposal.size()},
                {"eos",!output.empty() && llama_vocab_is_eog(vocab,output.back())},{"verification_mode","serial"}};
    }
    json verify(const tokens & prefix,tokens proposal,int remaining) {
        require(proposal.size()<=16 && remaining>=1 && remaining<=512,"invalid verification budget");
        valid(proposal);valid(prefix);require(!prefix.empty() && prefix.size()<size_t(nctx) && prefix.size()+proposal.size()<=size_t(nctx),"verification would exceed context");
        // A fully accepted proposal may add a bonus token. Clamp the output to
        // the actual context space instead of trusting the caller's budget.
        remaining=std::min(remaining,nctx-int(prefix.size()));
        // One verification pass: retain KV before the committed anchor and
        // evaluate [anchor, proposal...] together. No per-block logit repair pass.
        tokens before(prefix.begin(),prefix.end()-1);
        size_t keep=0;while(keep<before.size() && keep<cached.size() && before[keep]==cached[keep])++keep;
        if(keep<cached.size()) {
            if(!llama_memory_seq_rm(llama_get_memory(ctx.get()),0,int(keep),-1)){reset();keep=0;}
            cached.resize(keep);next=-1;
        }
        while(keep<before.size()) {size_t end=std::min(keep+size_t(512),before.size());decode(tokens(before.begin()+keep,before.begin()+end),false);keep=end;}
        tokens batch{prefix.back()};batch.insert(batch.end(),proposal.begin(),proposal.end());
        auto choices=decode(batch,true);tokens output;int accepted=0;
        if(proposal.empty())output.push_back(choices[0]);
        else {
            for(size_t i=0;i<proposal.size();++i) {
                auto expected=choices[i];
                if(proposal[i]!=expected){output.push_back(expected);break;}
                output.push_back(proposal[i]);++accepted;
                if(llama_vocab_is_eog(vocab,proposal[i]) || int(output.size())>=remaining)break;
                if(i+1==proposal.size())output.push_back(choices.back());
            }
        }
        if(output.size()>size_t(remaining))output.resize(remaining);
        for(size_t i=0;i<output.size();++i)if(llama_vocab_is_eog(vocab,output[i])){output.resize(i+1);break;}
        accepted=std::min(accepted,int(output.size()));
        // Defer sync of the committed tail until the next block. Stale speculative
        // KV is never sampled: every entry point synchronizes its explicit prefix.
        return {{"tokens",output},{"accepted",accepted},{"proposed",proposal.size()},
                {"eos",!output.empty() && llama_vocab_is_eog(vocab,output.back())},{"verification_mode","experimental_batch"}};
    }
    tokens tokenize(const std::string & text,bool add_special,bool parse_special) {
        int size=-llama_tokenize(vocab,text.data(),int(text.size()),nullptr,0,add_special,parse_special);
        require(size>=0,"tokenization size failed");tokens ids(size);
        int used=llama_tokenize(vocab,text.data(),int(text.size()),ids.data(),size,add_special,parse_special);
        require(used>=0,"tokenization failed");ids.resize(used);return ids;
    }
    void untrusted_content(const std::string & text) {
        require(text.find('\0')==std::string::npos,"chat content contains NUL");
        // Fail closed instead of disabling the template's legitimate controls.
        // Literal control spellings in user/assistant/system content are not yet
        // supported by this wrapper; plain text and the trusted template remain.
        require(tokenize(text,false,false)==tokenize(text,false,true),
                "chat content contains a reserved control-token spelling");
    }
    json timing(double total) const {
        return {{"schema_version",1},{"native_total_ms",total},{"native_sync_ms",sync_ms},
                {"native_after_sync_ms",std::max(0.0,total-sync_ms)},
                {"native_decode_ms",decode_ms},{"native_decode_calls",decode_calls},
                {"native_decode_tokens",decode_tokens},{"native_sync_tokens",sync_tokens},
                {"sync_includes_decode",true}};
    }
    json call(const json & req) {
        sync_ms=decode_ms=0;decode_calls=decode_tokens=sync_tokens=0;syncing=false;
        const auto op=req.at("op").get<std::string>();if(op=="load")return load(req);
        require(ctx && vocab,"no model loaded");
        if(op=="reset"){reset();return {{"reset",true}};}
        if(op=="tokenize") {
            auto text=req.at("text").get<std::string>();require(text.size()<=131072,"prompt too long");
            if(req.value("chat",false)) {
                const char * tmpl=llama_model_chat_template(model.get(),nullptr);require(tmpl,"model has no chat template");
                std::vector<std::string> roles,contents;
                if(req.contains("messages") && req["messages"].is_array()) {
                    require(!req["messages"].empty() && req["messages"].size()<=32,"invalid message count");
                    size_t bytes=0;
                    for(const auto & m:req["messages"]){
                        auto role=m.at("role").get<std::string>();auto content=m.at("content").get<std::string>();
                        require(role=="system"||role=="user"||role=="assistant","unsupported chat role");
                        bytes+=content.size();require(bytes<=131072,"chat messages too long");untrusted_content(content);
                        roles.push_back(role);contents.push_back(content);
                    }
                } else {untrusted_content(text);roles.push_back("user");contents.push_back(text);}
                std::vector<llama_chat_message> messages;
                for(size_t i=0;i<roles.size();++i)messages.push_back({roles[i].c_str(),contents[i].c_str()});
                int size=llama_chat_apply_template(tmpl,messages.data(),messages.size(),true,nullptr,0);require(size>0 && size<=262144,"unsupported or excessive chat template");
                std::vector<char> out(size+1);int used=llama_chat_apply_template(tmpl,messages.data(),messages.size(),true,out.data(),int(out.size()));
                require(used>=0 && used<int(out.size()),"chat template formatting failed");text.assign(out.data(),used);
            }
            auto ids=tokenize(text,req.value("add_special",true),req.value("chat",false)||req.value("parse_special",false));
            require(!ids.empty(),"tokenization produced no tokens");require(ids.size()<size_t(nctx),"prompt exceeds context");
            return {{"tokens",ids},{"tokenizer_fingerprint",fingerprint}};
        }
        if(op=="text") {
            auto ids=req.at("tokens").get<tokens>();valid(ids);std::string text;
            for(auto id:ids){char small[256];int n=llama_token_to_piece(vocab,id,small,sizeof(small),0,false);
                if(n<0){std::vector<char> big(-n);n=llama_token_to_piece(vocab,id,big.data(),int(big.size()),0,false);require(n>=0,"detokenization failed");text.append(big.data(),n);}else text.append(small,n);}
            std::string hex;const char * digits="0123456789abcdef";
            for(unsigned char c:text){hex+=digits[c>>4];hex+=digits[c&15];}
            return {{"text",text},{"bytes_hex",hex}};
        }
        auto prefix=req.at("prefix").get<tokens>();auto start=clock_type::now();
        if(op=="prefill"){sync(prefix);auto ms=elapsed(start);return {{"ms",ms},{"timing",timing(ms)}};}
        if(op=="draft"){auto ids=draft(prefix,req.value("count",4));auto ms=elapsed(start);return {{"tokens",ids},{"ms",ms},{"timing",timing(ms)},{"eos",!ids.empty()&&llama_vocab_is_eog(vocab,ids.back())},{"tokenizer_fingerprint",fingerprint}};}
        if(op=="verify"){
            auto proposal=req.at("proposal").get<tokens>();int remaining=req.value("remaining",512);
            auto value=req.value("strict",true)?verify_serial(prefix,proposal,remaining):verify(prefix,proposal,remaining);
            auto ms=elapsed(start);value["ms"]=ms;value["timing"]=timing(ms);return value;
        }
        throw std::runtime_error("unknown native operation");
    }
};

// Opaque IDs keep close/cancel safe even when the mobile lifecycle races an
// in-flight native call. A call retains the engine until it has returned.
static std::mutex handles_mutex;
static std::unordered_map<uintptr_t,std::shared_ptr<Engine>> handles;
static uintptr_t next_handle=1;
static std::shared_ptr<Engine> lookup(void * p) {
    std::lock_guard<std::mutex> lock(handles_mutex);auto it=handles.find(reinterpret_cast<uintptr_t>(p));
    return it==handles.end()?nullptr:it->second;
}
extern "C" void * nl_create() {
    try{auto engine=std::make_shared<Engine>();std::lock_guard<std::mutex> lock(handles_mutex);
        auto id=next_handle++;handles.emplace(id,std::move(engine));return reinterpret_cast<void *>(id);
    }catch(...){return nullptr;}
}
extern "C" void nl_cancel(void * p) {auto e=lookup(p);if(e)e->cancelled.store(true);}
#ifdef NODELIFT_TEST_FAULTS
static std::atomic<int> response_fault{0};
extern "C" void nl_test_fail_next_response(int kind) { response_fault.store(kind); }
#endif
extern "C" char * nl_call(void * p,const char * input) {
    // The outer boundary also contains allocations made while constructing an
    // error, serializing it and copying its result. Null is an allocation error.
    try {
        json result;
        try {
            auto retained=lookup(p);require(retained && input,"invalid native handle");auto & e=*retained;std::lock_guard<std::mutex> lock(e.mutex);
            e.cancelled.store(false);
            require(!e.closing.load(),"native handle is closing");
            bool loading=false;
            try {auto request=json::parse(input);loading=request.value("op",std::string())=="load";result=e.call(request);}
            catch (...) {
                if(loading){e.ctx.reset();e.model.reset();e.vocab=nullptr;e.cached.clear();e.next=-1;}
                else if(e.ctx)e.reset();
                throw;
            }
        }catch(const std::exception & ex){result={{"error",ex.what()}};}catch(...){result={{"error","unknown native error"}};}
#ifdef NODELIFT_TEST_FAULTS
        int fault=response_fault.exchange(0);if(fault==1)throw std::bad_alloc();if(fault==2)return nullptr;
#endif
        return strdup(result.dump(-1,' ',true,json::error_handler_t::replace).c_str());
    } catch (...) {return nullptr;}
}
extern "C" void nl_free_string(char * p) {free(p);}
extern "C" void nl_destroy(void * p) {
    std::lock_guard<std::mutex> lock(handles_mutex);
    auto it=handles.find(reinterpret_cast<uintptr_t>(p));
    if(it!=handles.end()){it->second->closing.store(true);it->second->cancelled.store(true);handles.erase(it);}
}
