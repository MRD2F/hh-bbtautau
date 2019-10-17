import numpy as np
import ROOT
import sys
ROOT.gInterpreter.Declare('#include "../include/AnaPyInterface.h"')
max_jet = 10


def DefineVariables(sample_name, parity) :
    df = ROOT.ROOT.RDataFrame('all_events', sample_name)

    if parity >= 0 and parity <= 1:
        df = df.Filter('evt % 2 == {}'.format(parity))

    df =  df.Define('deepFlavour_bVSall', 'MakeDeepFlavour_bVSall(jets_deepFlavour_b, jets_deepFlavour_bb, jets_deepFlavour_lepb)') \
            .Define('jets_deepFlavourOrderedIndex', 'CreateOrderedIndex(jets_p4, deepFlavour_bVSall, true, {})'.format(max_jet))  \
            .Define('jets_genbJet', 'MakeGenbJet(jets_genJetIndex, jets_deepFlavourOrderedIndex)') \
            .Filter('std::accumulate(jets_genbJet.begin(), jets_genbJet.end(), 0) == 2')


    df = df.Define('n_jets', 'jets_deepFlavourOrderedIndex.size()') \
           .Define('htt_p4', 'getHTTp4(lep_p4, lep_genTauIndex)') \
           .Define('htt_pt', 'htt_p4.pt()') \
           .Define('htt_eta', 'htt_p4.eta()') \
           .Define('htt_phi' , 'htt_p4.phi()') \
           .Define('met_pt', 'met_pt(pfMET_p4)') \
           .Define('rel_met_pt_htt_pt', 'rel_pt_met_htt(met_pt, htt_pt)') \
           .Define('htt_met_dphi', 'httMetDeltaPhi(htt_p4, pfMET_p4)')

    for n_jet in range(10):
        df = df.Define('jet_{}_valid'.format(n_jet), 'static_cast<float>({} < jets_deepFlavourOrderedIndex.size())'.format(n_jet)) \
               .Define('jet_{}_pt'.format(n_jet), 'jet_p4_pt(jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('jet_{}_eta'.format(n_jet), 'jet_p4_eta(jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('jet_{}_E'.format(n_jet), 'jet_p4_E(jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('jet_{}_M'.format(n_jet), 'jet_p4_M(jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('rel_jet_{}_M_pt'.format(n_jet), 'rel_jet_M_pt(jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('rel_jet_{}_E_pt'.format(n_jet), 'rel_jet_E_pt(jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('jet_{}_genbJet'.format(n_jet), 'jet_genbJet(jets_genJetIndex, jets_deepFlavourOrderedIndex, {}, jets_p4)'.format(n_jet)) \
               .Define('jet_{}_genJetIndex'.format(n_jet), 'jet_genJetIndex(jets_genJetIndex, {}, jets_p4)'.format(n_jet)) \
               .Define('jet_{}_deepFlavour'.format(n_jet), 'jets_deepFlavour(jets_deepFlavour_b,  jets_deepFlavour_bb, jets_deepFlavour_lepb, jets_deepFlavourOrderedIndex, {})'.format(n_jet)) \
               .Define('jet_{}_htt_dphi'.format(n_jet), 'httDeltaPhi_jet(htt_p4, jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet)) \
               .Define('jet_{}_htt_deta'.format(n_jet), 'httDeltaEta_jet(htt_p4, jets_deepFlavourOrderedIndex, jets_p4, {})'.format(n_jet))

    return df

def CreateColums() :
    evt_columns = [ 'sample_type', 'spin', 'mass_point', 'node', 'sample_year', 'channelId', 'htt_pt', 'htt_eta',
                    'htt_phi', 'htt_met_dphi', 'met_pt', 'rel_met_pt_htt_pt', 'n_jets'
    ]

    jet_column = [ 'jet_{}_valid', 'jet_{}_pt', 'jet_{}_eta', 'jet_{}_E', 'jet_{}_M', 'rel_jet_{}_M_pt', 'rel_jet_{}_E_pt',
                   'jet_{}_htt_deta', 'jet_{}_deepFlavour', 'jet_{}_htt_dphi', 'jet_{}_genbJet', 'jet_{}_genJetIndex'
    ]

    all_vars = evt_columns + jet_column
    jet_columns = []

    for jet_var in jet_column :
        for n in range(10) :
            jet_columns.append(jet_var.format(n))

    return evt_columns, jet_column, all_vars, jet_columns


def GetIndex(x) :
    evt_columns, jet_column, all_vars, jet_columns = CreateColums()
    if type(x) == list :
        all_indexes = []
        for var in range(len(x)) :
            var_name = x[var]
            idx = all_vars.index(var_name)
            all_indexes.append(idx)
        return all_indexes
    elif type(x) == str :
        idx = all_vars.index(x)
        return idx

def CreateInputs(raw_data):
    evt_columns, jet_column, all_vars, jet_columns = CreateColums()
    n_vars_evt = len(evt_columns)
    n_vars_jet = len(jet_column)
    max_jet = 10
    n_evt = len(raw_data['n_jets'])

    data = np.zeros((n_evt, max_jet, n_vars_evt+n_vars_jet ), dtype=np.float32)

    evt_vars_idx = GetIndex(evt_columns)
    jet_vars_idx = GetIndex(jet_column)

    for jet_idx in range(max_jet):
        for n in range(len(evt_vars_idx)):
            data[:, jet_idx, evt_vars_idx[n]] = raw_data[all_vars[evt_vars_idx[n]]][:]
        for n in range(len(jet_vars_idx)):
            data[:, jet_idx, jet_vars_idx[n]] = raw_data[all_vars[jet_vars_idx[n]].format(jet_idx)][:]
    return data

def CreateRootDF(sample_name, parity, do_shuffle):
    df = DefineVariables(sample_name, parity)
    evt_columns, jet_column, all_vars, jet_columns = CreateColums()
    data_raw = df.AsNumpy(columns=evt_columns+jet_columns)
    data = CreateInputs(data_raw)
    if do_shuffle:
        np.random.shuffle(data)

    return data

def CreateXY(data):
    training_evt_vars     = [ 'sample_year', 'channelId', 'htt_pt', 'htt_eta', 'htt_phi', 'htt_met_dphi', 'rel_met_pt_htt_pt']
    idx_training_evt_vars =  GetIndex(training_evt_vars)

    training_jet_vars     = [ 'jet_{}_valid', 'jet_{}_pt', 'jet_{}_eta', 'rel_jet_{}_M_pt',
                              'rel_jet_{}_E_pt','jet_{}_htt_deta', 'jet_{}_deepFlavour', 'jet_{}_htt_dphi' ]#, 'jet_{}_genJetIndex' ]
    idx_training_jet_vars =  GetIndex(training_jet_vars)

    training_vars         =   training_jet_vars + training_evt_vars
    training_vars_idx     =   idx_training_jet_vars + idx_training_evt_vars

    genTruth_var          = [ 'jet_{}_genbJet' ]
    idx_genTruth_var      =  GetIndex(genTruth_var)

    X = data[:, :, training_vars_idx]
    Y = data[:, :, idx_genTruth_var]

    var_pos = {}
    for n in range(len(training_vars)):
        var_pos[training_vars[n]] = n

    valid_pos = var_pos['jet_{}_valid']
    for jet_idx in range(X.shape[1]):
        for var_idx in range(X.shape[2]):
            X[:, jet_idx, var_idx] = X[:, jet_idx, var_idx] * X[:, jet_idx, valid_pos]

    return X,Y,var_pos
