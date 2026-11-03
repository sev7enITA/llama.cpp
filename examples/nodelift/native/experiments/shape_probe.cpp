// Research-only: compare one-token and batched llama.cpp evaluation from an
// identical, freshly prefetched prefix. This tool is not a production verifier.
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
using Tokens = std::vector<llama_token>;
using Clock = std::chrono::steady_clock;
static void require(bool ok,const char * msg){if(!ok)throw std::runtime_error(msg);}
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
using Context=std::unique_ptr<llama_context,decltype(&llama_free)>;
struct Ranked {int first=0,second=0;float margin=0;};
static Ranked rank(const float * p,int n){
 require(p && n>1,"missing logits"); int first=0,second=1;
 if(p[second]>p[first])std::swap(first,second);
 for(int i=2;i<n;++i){require(std::isfinite(p[i]),"non-finite logits");if(p[i]>p[first]){second=first;first=i;}else if(p[i]>p[second])second=i;}
 return {first,second,p[first]-p[second]};
}
static std::vector<std::vector<float>> eval(llama_context * ctx,const Tokens & ids,int position,bool all,int vocab){
 require(!ids.empty() && ids.size()<=512,"invalid evaluation window");
 auto batch=llama_batch_init(int(ids.size()),0,1);batch.n_tokens=int(ids.size());
 for(int i=0;i<batch.n_tokens;++i){batch.token[i]=ids[i];batch.pos[i]=position+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=all || i==batch.n_tokens-1;}
 int rc=llama_decode(ctx,batch);llama_batch_free(batch);require(rc==0,"llama_decode failed");
 std::vector<std::vector<float>> result;
 if(all)for(int i=0;i<int(ids.size());++i){auto p=llama_get_logits_ith(ctx,i);require(p,"missing per-token logits");result.emplace_back(p,p+vocab);}
 else {auto p=llama_get_logits_ith(ctx,-1);require(p,"missing final logits");result.emplace_back(p,p+vocab);}
 return result;
}
static void prefill(llama_context * ctx,const Tokens & ids,int vocab){
 llama_memory_clear(llama_get_memory(ctx),true);
 for(size_t start=0;start<ids.size();start+=512){auto end=std::min(start+size_t(512),ids.size());eval(ctx,Tokens(ids.begin()+start,ids.begin()+end),int(start),false,vocab);}
}
static std::vector<std::vector<float>> single(llama_context * ctx,const Tokens & window,int position,int vocab){
 std::vector<std::vector<float>> out;
 for(auto id:window){auto one=eval(ctx,Tokens{id},position++,false,vocab);out.push_back(std::move(one[0]));}
 return out;
}
int main(int argc,char **argv){try{
 require(argc>=7 && argc<=9,"usage: shape_probe MODEL CASES OUTPUT GPU_LAYERS THREADS FLASH(0/1) [N_CTX] [SERIAL_HISTORY(0/1)]");
 int gpu=std::stoi(argv[4]),threads=std::stoi(argv[5]);bool flash=std::stoi(argv[6])!=0;
 int nctx=argc>=8?std::stoi(argv[7]):512;bool serial_history=argc>=9 && std::stoi(argv[8])!=0;
 require(gpu>=0 && gpu<=999 && threads>=1 && threads<=16 && nctx>=256 && nctx<=8192,"invalid backend config");
 std::ifstream in(argv[2]);require(bool(in),"cannot open cases");json cases;in>>cases;
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=gpu;
 std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);require(bool(model),"model load failed");
 auto vocab=llama_model_get_vocab(model.get());int nv=llama_vocab_n_tokens(vocab);
 auto cp=llama_context_default_params();cp.n_ctx=nctx;cp.n_batch=512;cp.n_ubatch=512;cp.n_threads=threads;cp.n_threads_batch=threads;cp.flash_attn_type=flash?LLAMA_FLASH_ATTN_TYPE_ENABLED:LLAMA_FLASH_ATTN_TYPE_DISABLED;
 Context a(llama_init_from_model(model.get(),cp),llama_free), b(llama_init_from_model(model.get(),cp),llama_free), repeat(llama_init_from_model(model.get(),cp),llama_free);require(a&&b&&repeat,"context allocation failed");
 json output={{"gpu_layers_requested",gpu},{"threads",threads},{"flash_attention_requested",flash},{"n_ctx",nctx},{"serial_history",serial_history},{"runtime",llama_version()},{"windows",json::array()}};
 for(const auto & c:cases.at("cases")){
  int history_width=c.value("history_batch_width",1);require(history_width>=1 && history_width<=512,"invalid history width");
  Tokens prompt=c.at("prompt_tokens").get<Tokens>(),generated=c.at("generated_tokens").get<Tokens>();require(!prompt.empty(),"empty prompt");
  for(const auto & w:c.at("windows")){
   int offset=w.at("offset"),width=w.at("width");require(offset>=0 && width>=1 && size_t(offset+width-1)<=generated.size(),"invalid case window");
   Tokens before=prompt;before.insert(before.end(),generated.begin(),generated.begin()+offset);auto anchor=before.back();before.pop_back();
   Tokens block{anchor};block.insert(block.end(),generated.begin()+offset,generated.begin()+offset+width-1);
   require(before.size()+block.size()<=size_t(nctx),"window exceeds context");
   for(auto ctx:{a.get(),b.get(),repeat.get()}){
    if(serial_history && offset>0){
     prefill(ctx,prompt,nv);int step=ctx==b.get()?history_width:1;
     for(int at=0;at<offset-1;at+=step){int end=std::min(at+step,offset-1);eval(ctx,Tokens(generated.begin()+at,generated.begin()+end),int(prompt.size())+at,ctx==b.get() && step>1,nv);}
    }
    else prefill(ctx,before,nv);
   }
   auto t=Clock::now();auto serial=single(a.get(),block,int(before.size()),nv);double serialms=ms(t);
   t=Clock::now();auto batch=eval(b.get(),block,int(before.size()),true,nv);double batchms=ms(t);
   auto control=single(repeat.get(),block,int(before.size()),nv);
   json row={{"case",c.at("name")},{"history_batch_width",history_width},{"offset",offset},{"width",width},{"prefix_tokens",before.size()+1},{"serial_ms",serialms},{"batch_ms",batchms},{"positions",json::array()}};
   for(int j=0;j<width;++j){
    auto sr=rank(serial[j].data(),nv),br=rank(batch[j].data(),nv),cr=rank(control[j].data(),nv);float maxab=0,maxaa=0;
    for(int v=0;v<nv;++v){require(std::isfinite(serial[j][v])&&std::isfinite(batch[j][v])&&std::isfinite(control[j][v]),"non-finite logits");maxab=std::max(maxab,std::abs(serial[j][v]-batch[j][v]));maxaa=std::max(maxaa,std::abs(serial[j][v]-control[j][v]));}
    row["positions"].push_back({{"index",j},{"serial_top1",sr.first},{"batch_top1",br.first},{"control_top1",cr.first},{"serial_top2",sr.second},{"batch_top2",br.second},{"serial_margin",sr.margin},{"batch_margin",br.margin},{"max_abs_serial_batch",maxab},{"max_abs_serial_control",maxaa},{"top1_flipped",sr.first!=br.first},{"control_top1_flipped",sr.first!=cr.first}});
   }
   output["windows"].push_back(row);
  }
 }
 std::ofstream f(argv[3]);require(bool(f),"cannot open output");f<<output.dump(2)<<'\n';f.close();require(bool(f),"cannot write output");
 std::cout<<"wrote "<<output["windows"].size()<<" windows to "<<argv[3]<<"\n";
 return 0;
 }catch(const std::exception & e){std::cerr<<"shape_probe: "<<e.what()<<'\n';return 1;}}
