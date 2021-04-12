/*! Class than provides nonresonant EFT model for event analyzers.
This file is part of https://github.com/hh-italian-group/hh-bbtautau. */

#include "hh-bbtautau/Analysis/include/NonResModel.h"

#include "h-tautau/McCorrections/include/PileUpWeight.h"

namespace analysis {

NonResModel::ParamPositionDesc::ParamPositionDesc(const NameMap& names)
{
    SetParamPosition(names, "kl", kl);
    SetParamPosition(names, "kt", kt);
    SetParamPosition(names, "c2", c2);
    SetParamPosition(names, "cg", cg);
    SetParamPosition(names, "c2g", c2g);
}

NonResModel::Point NonResModel::ParamPositionDesc::CreatePoint(const ValueVec& param_values) const
{
    Point point;
    SetValue(param_values, "kl", kl, point.kl);
    SetValue(param_values, "kt", kt, point.kt);
    SetValue(param_values, "c2", c2, point.c2);
    SetValue(param_values, "cg", cg, point.cg);
    SetValue(param_values, "c2g", c2g, point.c2g);
    return point;
}

void NonResModel::ParamPositionDesc::SetParamPosition(const NameMap& names, const std::string& name, OptPos& pos)
{
    auto iter = names.find(name);
    if(iter != names.end())
        pos = iter->second;
}

void NonResModel::ParamPositionDesc::SetValue(const ValueVec& values, const std::string& name,
                                              const OptPos& pos, double& value)
{
    if(pos) {
        if(values.size() <= *pos)
            throw exception("Value not found for EFT parameter %1%.") % name;
        value = Parse<double>(values.at(*pos));
    }
}

NonResModel::NonResModel(const mc_corrections::EventWeights_HH& _weights, const SampleDescriptor& sample,
                         std::shared_ptr<TFile> file, tuple_skimmer::CrossSectionProvider& xs_provider) :
        weighting_mode(WeightType::PileUp, WeightType::BSM_to_SM, WeightType::GenEventWeight),
        weights(_weights),
        eft_weights(weights.GetProviderT<NonResHH_EFT::WeightProvider>(WeightType::BSM_to_SM)),
        points_are_orthogonal(sample.create_orthogonal_points)
{
    const ParamPositionDesc param_positions(sample.GetModelParameterNames());
    eft_weights->AddFile(*file);
    eft_weights->CreatePdfs();

    if(weighting_mode.count(WeightType::PileUp)){
        auto pile_up_weight = weights.GetProviderT<mc_corrections::PileUpWeightEx>(mc_corrections::WeightType::PileUp);
        pile_up_weight->SetActiveDataset(sample.reference_pu_sample);
    }

    for(const auto& sample_wp : sample.working_points) {
        point_names.push_back(sample_wp.full_name);
        points.push_back(param_positions.CreatePoint(sample_wp.param_values));
        double xs = -1;
        if(!sample_wp.cross_section.empty())
            xs = xs_provider.GetCrossSection(sample_wp.cross_section);
        point_xs.push_back(xs);
    }
    total_shape_weights = weights.GetTotalShapeWeights(file, weighting_mode, points, sample.create_orthogonal_points);
}

void NonResModel::ProcessEvent(const EventAnalyzerDataId& anaDataId, EventInfo& event, double weight,
                               double shape_weight, bbtautau::AnaTupleWriter::DataIdMap& dataIds, double cross_section,
                               std::map<UncertaintySource, std::map<UncertaintyScale, float>>& uncs_weight_map,
                               std::map<int, double>& weights_bench)
{


    std::map<std::string, int> eft_point_names = {
        {"GluGluSignal_NonRes_benchScan_kl7.5_kt1_c2-1_cg0_c2g0", 1},
        {"GluGluSignal_NonRes_benchScan_kl1_kt1_c20.5_cg-0.8_c2g0.6", 2},
        {"GluGluSignal_NonRes_benchScan_kl1_kt1_c2-1.5_cg0_c2g-0.8", 3},
        {"GluGluSignal_NonRes_benchScan_kl-3.5_kt1.5_c2-3_cg0_c2g0", 4},
        {"GluGluSignal_NonRes_benchScan_kl1_kt1_c20_cg0.8_c2g-1", 5},
        {"GluGluSignal_NonRes_benchScan_kl2.4_kt1_c20_cg0.2_c2g-0.2", 6},
        {"GluGluSignal_NonRes_benchScan_kl5_kt1_c20_cg0.2_c2g-0.2", 7},
        {"GluGluSignal_NonRes_benchScan_kl15_kt1_c20_cg-1_c2g1", 8},
        {"GluGluSignal_NonRes_benchScan_kl1_kt1_c21_cg-0.6_c2g0.6", 9},
        {"GluGluSignal_NonRes_benchScan_kl10_kt1.5_c2-1_cg0_c2g0", 10},
        {"GluGluSignal_NonRes_benchScan_kl2.4_kt1_c20_cg1_c2g-1", 11},
        {"GluGluSignal_NonRes_benchScan_kl15_kt1_c21_cg0_c2g0", 12},
        {"GluGluSignal_NonRes_benchScan_kl0_kt1_c20_cg0_c2g0", 13},
        {"GluGluSignal_NonRes_benchScan_kl1_kt1_c20_cg0_c2g0", 14},
    };


    // std::cout << "n size = " << points.size() << "\n";
    for(size_t n = 0; n < points.size(); ++n) {
        if(points_are_orthogonal && (event->evt % points.size()) != n) continue;
        const auto final_id = anaDataId.Set(point_names.at(n));
        eft_weights->SetTargetPoint(points.at(n));
        const double eft_weight = eft_weights->Get(event);
        const double shape_weight_correction = shape_weight / total_shape_weights.at(UncertaintyScale::Central).at(n);
        const double eft_weight_correction = eft_weight / event->weight_bsm_to_sm;
        const double xs_correction = point_xs.at(n) > 0 ? point_xs.at(n) / cross_section : 1.;
        const double final_weight = weight * shape_weight_correction * eft_weight_correction * xs_correction;
        dataIds[final_id] = final_weight;

        // std::cout << "point: " << eft_point_names.at(point_names.at(n)) << ", point name: " <<  point_names.at(n)
        //           << ", weight: " << shape_weight_correction * eft_weight_correction * xs_correction << "\n";
        std::cout << "eft_point_names.at(point_names.at(n)) : " << eft_point_names.at(point_names.at(n)) << "\n";
        std::cout << "point_names.at(n) : " << point_names.at(n)  << "\n";
        std::cout << "final_weight : "<< final_weight << "\n";
        weights_bench[eft_point_names[point_names.at(n)]] = final_weight;
        std::cout << "size map at NonRes: "<< weights_bench.size() << "\n";
        // std::cout< "point_names: " << point_names.at(n) << "\n";

        uncs_weight_map[UncertaintySource::PileUp][UncertaintyScale::Central] = static_cast<float>(event->weight_pu /
                total_shape_weights.at(UncertaintyScale::Central).at(n));
        uncs_weight_map[UncertaintySource::PileUp][UncertaintyScale::Up] = static_cast<float>(event->weight_pu_up /
                total_shape_weights.at(UncertaintyScale::Up).at(n));
        uncs_weight_map[UncertaintySource::PileUp][UncertaintyScale::Down] = static_cast<float>(event->weight_pu_down /
                total_shape_weights.at(UncertaintyScale::Down).at(n));
    }
}

} // namespace analysis
