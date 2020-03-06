/*! Produce synchronization tree.
This file is part of https://github.com/hh-italian-group/h-tautau. */

#include "AnalysisTools/Run/include/program_main.h"
#include "AnalysisTools/Core/include/RootExt.h"
#include "AnalysisTools/Core/include/EnumNameMap.h"
#include "AnalysisTools/Core/include/AnalysisMath.h"
#include "AnalysisTools/Core/include/TextIO.h"
#include "h-tautau/Analysis/include/EventInfo.h"
#include "h-tautau/Core/include/AnalysisTypes.h"
#include "h-tautau/Cuts/include/Btag_2016.h"
#include "h-tautau/Cuts/include/Btag_2017.h"
#include "h-tautau/McCorrections/include/EventWeights.h"
#include "hh-bbtautau/Analysis/include/SampleDescriptorConfigEntryReader.h"
#include "hh-bbtautau/Analysis/include/SyncTupleHTT.h"
#include "h-tautau/Analysis/include/SignalObjectSelector.h"
#include "AnalysisTools/Core/include/EventIdentifier.h"
// #include "h-tautau/Analysis/include/EventCandidate.h"

struct Arguments {
    REQ_ARG(std::string, mode);
    REQ_ARG(std::string, input_file);
    REQ_ARG(std::string, tree_name);
    REQ_ARG(std::string, period);
    REQ_ARG(std::string, trigger_cfg);
    REQ_ARG(std::string, output_file);
    REQ_ARG(bool, apply_trigger_vbf);
    REQ_ARG(bool, isData);
    OPT_ARG(std::string, mva_setup, "");
    OPT_ARG(bool, fill_tau_es_vars, false);
    OPT_ARG(bool, fill_jet_es_vars, false);
    OPT_ARG(std::string, jet_unc_source, "");
    OPT_ARG(std::string, jet_uncertainty, "");
};

namespace analysis {

enum class SyncMode { HTT, HH };

ENUM_NAMES(SyncMode) = {
    { SyncMode::HTT, "htt" },
    { SyncMode::HH, "hh" }
};

class SyncTreeProducer {
public:
    using Event = ntuple::Event;
    using EventTuple = ntuple::EventTuple;
    using SyncEvent = htt_sync::SyncEvent;
    using SyncTuple = htt_sync::SyncTuple;

    static constexpr float default_value = std::numeric_limits<float>::lowest();
    // static constexpr int default_int_value = std::numeric_limits<int>::lowest();

    SyncTreeProducer(const Arguments& _args) : args(_args), syncMode(Parse<SyncMode>(args.mode())),
                                                            run_period(Parse<analysis::Period>(args.period())),
                                                            signalObjectSelector(ConvertMode(syncMode))
                                                            // eventWeights(Parse<analysis::Period>(args.period()), JetOrdering::DeepCSV, DiscriminatorWP::Medium, true),

    {
        if(args.mva_setup().size()) {
            ConfigReader config_reader;

            MvaReaderSetupCollection mva_setup_collection;
            MvaReaderSetupEntryReader mva_entry_reader(mva_setup_collection);
            config_reader.AddEntryReader("MVA", mva_entry_reader, true);
            config_reader.ReadConfig(args.mva_setup());

            std::vector<MvaReaderSetup> mva_setups;
            for(const auto& mva_setup_element : mva_setup_collection) {
                mva_setups.push_back(mva_setup_element.second);
            }
            mva_setup = mva_setups.size() == 1 ? mva_setups.front() : MvaReaderSetup::Join(mva_setups);

            mva_reader = std::make_shared<analysis::mva_study::MvaReader>();
            InitializeMvaReader();
        }
        // EventCandidate::InitializeUncertainties(run_period, false, ".",
        //                                         TauIdDiscriminator::byDeepTau2017v2p1VSjet,
        //                                         TauIdDiscriminator::byDeepTau2017v2p1VSe);

    }

    void Run()
    {
        std::cout << boost::format("Processing input file '%1%' into output file '%2%' using %3% mode.\n")
                   % args.input_file() % args.output_file() % args.mode();

        std::map<std::string,std::pair<std::shared_ptr<ntuple::EventTuple>,Long64_t>> map_event;

        auto originalFile = root_ext::OpenRootFile(args.input_file());
        auto outputFile = root_ext::CreateRootFile(args.output_file());
        auto originalTuple = ntuple::CreateEventTuple("events",originalFile.get(),true,ntuple::TreeState::Full);
        const Long64_t n_entries = originalTuple->GetEntries();

        SyncTuple sync(args.tree_name(), outputFile.get(), false);
        auto summaryTuple = ntuple::CreateSummaryTuple("summary", originalFile.get(), true, ntuple::TreeState::Full);
        summaryTuple->GetEntry(0);
        SummaryInfo summaryInfo(summaryTuple->data(), Parse<Channel>(args.tree_name()), args.trigger_cfg());
        std::cout << "n_entries " << n_entries << '\n';

        for(Long64_t current_entry = 0; current_entry < n_entries; ++current_entry) {
            originalTuple->GetEntry(current_entry);
            if(static_cast<Channel>((*originalTuple)().channelId) == Channel::MuMu){ //temporary fix due tue a bug in mumu channel in production
                    (*originalTuple)().first_daughter_indexes = {0};
                    (*originalTuple)().second_daughter_indexes = {1};
            }
            const ntuple::Event& event = (*originalTuple).data();
            if(ToString(static_cast<Channel>(event.channelId))  != args.tree_name()) continue;

            // const EventIdentifier EventId(event.run, event.lumi, event.evt);
            // const EventIdentifier EventIdTest(1,1681,263510);
            //
            // if(EventId == EventIdTest){
            //     std::cout << "Pippo" << "\n";
            // }
            // if(!(EventId == EventIdTest)) continue;
            // std::cout << event.run << "," << event.lumi << ","<<  event.evt << "\n";
          // std::cout << "n_entries"  << '\n';
            FillSyncTuple(sync, event, summaryInfo);
        }
        sync.Write();
    }

private:

    void InitializeMvaReader()
    {
        using MvaKey = mva_study::MvaReader::MvaKey;
        if(!mva_setup.is_initialized())
            throw analysis::exception("Mva setup is not initialized.");
        for(const auto& method : mva_setup->trainings) {
            const auto& name = method.first;
            const auto& file = method.second;
            const auto& vars = mva_setup->variables.at(name);
            const auto& masses = mva_setup->masses.at(name);
            const auto& spins = mva_setup->spins.at(name);
            const bool legacy = mva_setup->legacy.count(name);
            const bool legacy_lm = legacy && mva_setup->legacy.at(name) == "lm";
            const size_t n_wp = masses.size();
            for(size_t n = 0; n < n_wp; ++n) {
                const MvaKey key{name, static_cast<int>(masses.at(n)), spins.at(n)};
                mva_reader->Add(key, file, vars, legacy, legacy_lm);
            }
        }
    }

    SignalMode ConvertMode(SyncMode syncMode)
    {
        std::map<SyncMode,SignalMode> signalMode_map ={
            {SyncMode::HTT, SignalMode::HTT},
            {SyncMode::HH, SignalMode::HH}
        };
        return signalMode_map.at(syncMode);
    }

    void FillSyncTuple(SyncTuple& sync, const ntuple::Event& event,const SummaryInfo& summaryInfo) const
    {
        static const std::map<std::pair<Period, Channel>, std::vector<std::string>> triggerPaths = {
            { { Period::Run2016, Channel::ETau }, { "HLT_Ele25_eta2p1_WPTight_Gsf_v" } },
            { { Period::Run2016, Channel::MuTau }, { "HLT_IsoMu22_v", "HLT_IsoMu22_eta2p1_v", "HLT_IsoTkMu22_v",
                                "HLT_IsoTkMu22_eta2p1_v", "HLT_IsoMu19_eta2p1_LooseIsoPFTau20_v",
                                "HLT_IsoMu19_eta2p1_LooseIsoPFTau20_SingleL1_v" } },
            { { Period::Run2016, Channel::TauTau }, { "HLT_DoubleMediumIsoPFTau35_Trk1_eta2p1_Reg_v",
                                "HLT_DoubleMediumCombinedIsoPFTau35_Trk1_eta2p1_Reg_v" } },
            { { Period::Run2016, Channel::MuMu }, { "HLT_IsoMu22_v" } },
            { { Period::Run2017, Channel::ETau }, { "HLT_Ele32_WPTight_Gsf_v", "HLT_Ele35_WPTight_Gsf_v",
                                "HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1_v",
                                "HLT_Ele32_WPTight_Gsf_L1DoubleEG" } },
            { { Period::Run2017, Channel::MuTau }, { "HLT_IsoMu24_v", "HLT_IsoMu27_v",
                                "HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1_v" } },
            { { Period::Run2017, Channel::TauTau }, { "HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_v",
                                "HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg_v",
                                "HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg_v" } },
            { { Period::Run2017, Channel::MuMu }, { "HLT_IsoMu24_v", "HLT_IsoMu27_v" } },
            { { Period::Run2018, Channel::ETau }, { "HLT_Ele32_WPTight_Gsf_v", "HLT_Ele35_WPTight_Gsf_v",
                                "HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTau30_eta2p1_CrossL1_v",
                                "HLT_Ele24_eta2p1_WPTight_Gsf_LooseChargedIsoPFTauHPS30_eta2p1_CrossL1_v" } },
            { { Period::Run2018, Channel::MuTau }, { "HLT_IsoMu24_v", "HLT_IsoMu27_v",
                                "HLT_IsoMu20_eta2p1_LooseChargedIsoPFTau27_eta2p1_CrossL1_v",
                                "HLT_IsoMu20_eta2p1_LooseChargedIsoPFTauHPS27_eta2p1_CrossL1_v" } },
            { { Period::Run2018, Channel::TauTau }, { "HLT_DoubleTightChargedIsoPFTau35_Trk1_TightID_eta2p1_Reg_v",
                                 "HLT_DoubleMediumChargedIsoPFTau40_Trk1_TightID_eta2p1_Reg_v",
                                 "HLT_DoubleTightChargedIsoPFTau40_Trk1_eta2p1_Reg_v",
                                 "HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v" } },
            { { Period::Run2018, Channel::MuMu }, { "HLT_IsoMu24_v", "HLT_IsoMu27_v" } },
        };

        static const std::map<std::pair<Period, Channel>, std::vector<std::string>> trigger_patterns_vbf = {
            { { Period::Run2017, Channel::TauTau }, {"HLT_VBF_DoubleLooseChargedIsoPFTau20_Trk1_eta2p1_Reg_v" } }
        };
        // std::cout << "1" << "\n";
        static const JetOrdering jet_ordering = JetOrdering::DeepFlavour;
        const Channel channel = Parse<Channel>(args.tree_name());
        const auto trig_key = std::make_pair(run_period, channel);

        // EventCandidate::InitializeUncertainties(run_period, false, ".",
        //                                         signalObjectSelector.GetTauVSjetDiscriminator().first,
        //                                         signalObjectSelector.GetTauVSeDiscriminator(Parse<Channel>(args.tree_name())).first);
        auto event_info = CreateEventInfo(event,signalObjectSelector,&summaryInfo,run_period,jet_ordering, true);
        // std::cout << "2" << "\n";
        if(!event_info.is_initialized()) return;
        // std::cout << "3" << "\n";
        // const auto& trig_list = triggerPaths.at(trig_key);
        // for(const auto& trig : trig_list) {
        //     const std::vector<std::string> single_trig = {trig};
        //     std::cout << trig << ": " << event_info->GetTriggerResults().AnyAcceptAndMatchEx(single_trig,
        //                                                             event_info->GetFirstLeg().GetMomentum().pt(),
        //                                                             event_info->GetSecondLeg().GetMomentum().pt())
        //             << std::endl;
        // }
        if(!event_info->GetTriggerResults().AnyAcceptAndMatchEx(triggerPaths.at(trig_key),
                                                                event_info->GetFirstLeg().GetMomentum().pt(),
                                                                event_info->GetSecondLeg().GetMomentum().pt())) return;
        if(syncMode == SyncMode::HH && !event_info->HasBjetPair()) return;
        if(syncMode == SyncMode::HH && !signalObjectSelector.PassLeptonVetoSelection(event)) return;
        if(syncMode == SyncMode::HH && !signalObjectSelector.PassMETfilters(event,run_period,args.isData())) return;

        if(args.apply_trigger_vbf() && trigger_patterns_vbf.count(trig_key)) {
            const auto first_vbf_jet = event_info->GetVBFJet(1);
            const auto second_vbf_jet = event_info->GetVBFJet(2);

            std::vector<boost::multiprecision::uint256_t> jet_trigger_match = {
                first_vbf_jet->triggerFilterMatch(),
                second_vbf_jet->triggerFilterMatch()
            };
            if(syncMode == SyncMode::HH
                    && !event_info->GetTriggerResults().AnyAcceptAndMatchEx(trigger_patterns_vbf.at(trig_key),
                            event_info->GetFirstLeg().GetMomentum().pt(), event_info->GetSecondLeg().GetMomentum().pt(),
                            jet_trigger_match))
                return;
        }

        htt_sync::FillSyncTuple(*event_info, sync, run_period, false, 1,
                                mva_reader.get(), nullptr, nullptr, nullptr, nullptr);
    }

private:
    Arguments args;
    SyncMode syncMode;
    analysis::Period run_period;
    // mc_corrections::EventWeights eventWeights;
    boost::optional<MvaReaderSetup> mva_setup;
    std::shared_ptr<analysis::mva_study::MvaReader> mva_reader;
    SignalObjectSelector signalObjectSelector;
};

} // namespace analysis

PROGRAM_MAIN(analysis::SyncTreeProducer, Arguments)
