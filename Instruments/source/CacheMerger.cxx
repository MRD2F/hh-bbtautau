/*! Merge multiple CacheTuples files into a single file.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#include <iostream>

#include "AnalysisTools/Core/include/RootExt.h"
#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/TextIO.h"
#include "h-tautau/Core/include/CacheTuple.h"
#include "h-tautau/Core/include/EventTuple.h"
#include "h-tautau/Analysis/include/EventCacheProvider.h"


struct Arguments {
    REQ_ARG(std::vector<std::string>, inputs);
    REQ_ARG(std::string, outputFile);
    REQ_ARG(std::string, channel);
};

namespace analysis {

class CacheMerger {
public:
    CacheMerger(const Arguments& _args): args(_args), output(root_ext::CreateRootFile(args.outputFile()))
    {
    }
    void Run(){
        std::vector<std::shared_ptr<TFile>> all_cache;
        for (size_t n = 0; n < args.inputs().size(); ++n){
            std::cout << args.inputs().at(n) << "\n";
            all_cache.push_back(root_ext::OpenRootFile(args.inputs().at(n)));
        }

        std::vector<std::shared_ptr<cache_tuple::CacheTuple>> cacheTuples;

        for (size_t n = 0; n < all_cache.size(); ++n){
            auto cacheFile = all_cache.at(n);
            try {
                auto cacheTuple = std::make_shared<cache_tuple::CacheTuple>(args.channel(), cacheFile.get(), true);
                cacheTuples.push_back(cacheTuple);
            } catch(std::exception&) {
                std::cerr << "WARNING: tree  << treeName" << " not found in file '"
                          << cacheFile << "'." << std::endl;
                cacheTuples.push_back(nullptr);
            }
        }
        cache_tuple::CacheTuple cache_out(args.channel(), output.get(), false);

        const Long64_t n_entries = cacheTuples.at(0)->GetEntries();

        for (int i = 0; i < cacheTuples.size() ; ++i){
            if(cacheTuples.at(0)->GetEntries() != cacheTuples.at(i)->GetEntries())
                throw exception ("The cache ntuple number '%1%' has an incorrect number of events '%2%' ('%3%').") % i
                                %cacheTuples.at(i)->GetEntries() %cacheTuples.at(0)->GetEntries();
        }

         for(Long64_t current_entry = 0; current_entry < n_entries; ++current_entry) {
             cache_out.GetEntry(current_entry);
             const cache_tuple::CacheEvent& c_out = cache_out.data();
             auto event_ptr = std::make_shared<cache_tuple::CacheEvent>(c_out);

             EventCacheProvider eventCacheProvider(*event_ptr);

             for (int i = 0; i < cacheTuples.size(); ++i){
                 cacheTuples.at(i)->GetEntry(current_entry);
                 const cache_tuple::CacheEvent& cache_event = cacheTuples.at(i)->data();
                 cache_out.Fill();
                 eventCacheProvider.AddEvent(cache_event);
            }
            eventCacheProvider.FillEvent(*event_ptr);
         }
         cache_out.Write();
         // std::cout << "final entries =" << cache_out.GetEntries()  << "\n";
    }
private:
    Arguments args;
    std::shared_ptr<TFile> output;

};
}// namespace analysis

PROGRAM_MAIN(analysis::CacheMerger, Arguments)
