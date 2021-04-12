#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "h-tautau/Core/include/EventTuple.h"
#include "h-tautau/JetTools/include/BTagger.h"

#include "../include/EventAnalyzerCore.h"



// #include "h-tautau/McCorrections/include/LeptonWeights.h"
// #include "h-tautau/McCorrections/include/BTagWeight.h"
// #include "h-tautau/McCorrections/include/GenEventWeight.h"
// #include "../include/DYModel.h"
// #include "../include/EventAnalyzerCore.h"

#include "h-tautau/McCorrections/include/LeptonWeights.h"
#include "h-tautau/McCorrections/include/BTagWeight.h"
#include "hh-bbtautau/McCorrections/include/EventWeights_HH.h"
#include "h-tautau/Analysis/include/SignalObjectSelector.h"
#include "h-tautau/McCorrections/include/GenEventWeight.h"

// #include "AnalysisTools/Core/include/AnalyzerData.h"
// #include "AnalysisTools/Core/include/EventIdentifier.h"
#include "h-tautau/Core/include/AnalysisTypes.h"
// #include "h-tautau/Analysis/src/EventInfo.cpp"

#include "hh-bbtautau/McCorrections/include/EventWeights_HH.h"
#include "../include/NonResModel2.h"
// #include <boost/regex.hpp>

struct Arguments {
    REQ_ARG(std::string, input_file);
    REQ_ARG(analysis::Channel, channel);
    REQ_ARG(analysis::Period, period);

    // REQ_ARG(std::string, outputFile);
    // OPT_ARG(bool, debug, false);
    // OPT_ARG(std::string, eventIdBranches, "run:lumi:evt");
};

namespace analysis {
using Event = ntuple::Event;

class NonResWeights {
public:
    NonResWeights(const Arguments& _args) :
        args(_args)//, output(root_ext::CreateRootFile(args.outputFile()))//, anaData(output), canvas("","", 600, 600)
    {
        bTagger = std::make_shared<BTagger>(args.period(), BTaggerKind::HHbtag);
        crossSectionProvider = std::make_shared<tuple_skimmer::CrossSectionProvider>("hh-bbtautau/Instruments/config/cross_section.cfg");

        // ConfigReader config_reader;
        //
        // SampleDescriptorConfigEntryReader sample_entry_reader(sample_descriptors);
        // config_reader.AddEntryReader("SAMPLE", sample_entry_reader, true);
    }

    void Run()
    {

        auto file = root_ext::OpenRootFile(args.input_file());
        auto tuple = ntuple::CreateEventTuple(ToString(args.channel()), file.get(), true, ntuple::TreeState::Skimmed);
        // auto summary_tuple = ntuple::CreateSummaryTuple("summary", file.get(), true,
                                                           // ntuple::TreeState::Skimmed);
        // const auto prod_summary = ntuple::MergeSummaryTuple(*summary_tuple);
        std::cout << "Initializing bTagger... " << std::endl;

        std::cout << "Creating event weights... " << std::endl;
        eventWeights_HH = std::make_shared<mc_corrections::EventWeights_HH>(args.period(), *bTagger);


        std::cout << "\t\tpreparing NonResModel... ";
        // const std::string& sample_name = "GluGluSignal_NonRes";
        // if(!sample_descriptors.count(sample_name))
        //     throw exception("Sample '%1%' not found while processing.") % sample_name;
        //
        // SampleDescriptor& sample = sample_descriptors.at(sample_name);
        // //        std::cout.flush();
        //        if(!crossSectionProvider)
        //            throw exception("path to the cross section config should be specified.");
        //        nonResModel = std::make_shared<NonResModel>(*eventWeights_HH, sample, file, *crossSectionProvider);
        //        std::cout << "done." << std::endl;
    }

protected:
Arguments args;
std::shared_ptr<mc_corrections::EventWeights_HH> eventWeights_HH;

private:
SampleDescriptorCollection sample_descriptors;
CombinedSampleDescriptorCollection cmb_sample_descriptors;
std::shared_ptr<BTagger> bTagger;
std::shared_ptr<tuple_skimmer::CrossSectionProvider> crossSectionProvider;


//std::shared_ptr<TFile> output;
};


} // namespace analysis
PROGRAM_MAIN(analysis::NonResWeights, Arguments)
