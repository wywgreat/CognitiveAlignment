#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import os
import csv
import numpy as np
from scipy.stats import spearmanr
from scipy.special import rel_entr  # 用于计算K-L散度
import matplotlib.pyplot as plt


def load_concept_cat_dic(file_path):

    df = pd.read_excel (file_path)


    concept_type_dict = {}
    concept_super_category_dict = {}
    concept_category_dict = {}


    for index, row in df.iterrows ():
        concept = row['Word']
        concept_type = row['Type']
        super_category = row['Super Category']
        category = row['Category']

        concept_type_dict[concept] = concept_type
        concept_super_category_dict[concept] = super_category
        concept_category_dict[concept] = category

    return concept_type_dict, concept_category_dict, concept_super_category_dict

def load_dim_domain_dic():
    dim_domain_dic = {
        "Vision": "Vision", "Bright": "Vision", "Dark": "Vision", "Color": "Vision", "Pattern": "Vision",
        "Large": "Vision", "Small": "Vision", "Motion": "Vision", "Biomotion": "Vision", "Fast": "Vision",
        "Slow": "Vision", "Shape": "Vision", "Complexity": "Vision", "Face": "Vision", "Body": "Vision",
        "Touch": "Somatic", "Temperature": "Somatic", "Texture": "Somatic", "Weight": "Somatic", "Pain": "Somatic",
        "Audition": "Audition", "Loud": "Audition", "Low": "Audition", "High": "Audition", "Sound": "Audition",
        "Music": "Audition", "Speech": "Audition", "Taste": "Gustation", "Smell": "Olfaction", "Head": "Motor",
        "UpperLimb": "Motor", "LowerLimb": "Motor", "Practice": "Motor", "Landmark": "Spatial", "Path": "Spatial",
        "Scene": "Spatial", "Near": "Spatial", "Toward": "Spatial", "Away": "Spatial", "Number": "Spatial",
        "Time": "Temporal", "Duration": "Temporal", "Long": "Temporal", "Short": "Temporal", "Caused": "Causal",
        "Consequential": "Causal", "Social": "Social", "Human": "Social", "Communication": "Social", "Self": "Social",
        "Cognition": "Cognition", "Benefit": "Emotion", "Harm": "Emotion", "Pleasant": "Emotion",
        "Unpleasant": "Emotion",
        "Happy": "Emotion", "Sad": "Emotion", "Angry": "Emotion", "Disgusted": "Emotion", "Fearful": "Emotion",
        "Surprised": "Emotion", "Drive": "Drive", "Needs": "Drive", "Attention": "Attention", "Arousal": "Attention"
    }

    domain_dimlst_dic = {
        "Vision": ["Vision", "Bright", "Dark", "Color", "Pattern", "Large", "Small", "Motion", "Biomotion", "Fast",
                   "Slow", "Shape", "Complexity", "Face", "Body"],
        "Somatic": ["Touch", "Temperature", "Texture", "Weight", "Pain"],
        "Audition": ["Audition", "Loud", "Low", "High", "Sound", "Music", "Speech"],
        "Gustation": ["Taste"],
        "Olfaction": ["Smell"],
        "Motor": ["Head", "UpperLimb", "LowerLimb", "Practice"],
        "Spatial": ["Landmark", "Path", "Scene", "Near", "Toward", "Away", "Number"],
        "Temporal": ["Time", "Duration", "Long", "Short"],
        "Causal": ["Caused", "Consequential"],
        "Social": ["Social", "Human", "Communication", "Self"],
        "Cognition": ["Cognition"],
        "Emotion": ["Benefit", "Harm", "Pleasant", "Unpleasant", "Happy", "Sad", "Angry", "Disgusted", "Fearful",
                    "Surprised"],
        "Drive": ["Drive", "Needs"],
        "Attention": ["Attention", "Arousal"]
    }

    return dim_domain_dic, domain_dimlst_dic


def kl_divergence(p, q):

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)


    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)


    p /= np.sum(p)
    q /= np.sum(q)


    return np.sum(rel_entr(p, q))


def export_to_excel(data, file_name, columns_lst, sheet_name='Sheet1'):

    df = pd.DataFrame(data, columns=columns_lst)
    df.to_excel(file_name, index=False, sheet_name=sheet_name)


def calculate_concept_correlations_kl_mse(human_data, model_data, focus_dims):
    print(focus_dims)
    corr_kl_mse_data = []
    for concept in human_data.index:
        if concept in model_data.index:
            human_values = human_data.loc[concept, focus_dims].values
            model_values = model_data.loc[concept, focus_dims].values

            correlation, _ = spearmanr(human_values, model_values)
            mse = np.mean((human_values - model_values) ** 2)
            kl = kl_divergence(human_values, model_values)
            corr_kl_mse_data.append((concept, correlation, mse, kl))
    return corr_kl_mse_data


def plot_LLM_human_radar(human_data, LLM_data, dims_lst, LLM_name):
    radar_folder = '../results/BBSR_ALLmodels_Radar'
    if not os.path.exists(radar_folder):
        os.makedirs(radar_folder)
    human_dims_mean_vec = []
    LLM_dims_mean_vec = []
    for each_dim in dims_lst:
        human_mean_dim = human_data[each_dim].mean()
        llm_mean_dim = LLM_data[each_dim].mean()
        human_dims_mean_vec.append(human_mean_dim)
        LLM_dims_mean_vec.append(llm_mean_dim)
    human_dims_mean_vec = np.array(human_dims_mean_vec)
    LLM_dims_mean_vec = np.array(LLM_dims_mean_vec)


    angles = np.linspace(0, 2 * np.pi, len(dims_lst), endpoint=False).tolist()

    human_dims_mean_vec = np.concatenate((human_dims_mean_vec, [human_dims_mean_vec[0]]))
    LLM_dims_mean_vec = np.concatenate((LLM_dims_mean_vec, [LLM_dims_mean_vec[0]]))
    angles += angles[:1]


    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, human_dims_mean_vec, color='blue', alpha=0.25, label='Human')
    ax.fill(angles, LLM_dims_mean_vec, color='red', alpha=0.25, label=LLM_name)
    ax.plot(angles, human_dims_mean_vec, color='blue', linewidth=2)
    ax.plot(angles, LLM_dims_mean_vec, color='red', linewidth=2)


    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims_lst)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.title('Radar Chart of human and '+LLM_name)
    plt.savefig(radar_folder+"/BBSR_human_vs_"+LLM_name+".png")


def BBSR_analysis(LLM_name, LLM_MEAN_file):
    print("Analysing " + LLM_name + " ...")
    analysis_data_folder = '../results/BBSR_' + LLM_name + '/processed/'
    analysis_result_folder = '../results/BBSR_' + LLM_name + '/statistics/'
    if not os.path.exists(analysis_result_folder):
        os.makedirs(analysis_result_folder)

    BBSR_file = "../data/[BBSR]WordSet1_Ratings.xlsx"
    dim_domain_dic, domain_dimlst_dic = load_dim_domain_dic()

    all65dims_lst = list(dim_domain_dic.keys())
    print(len(all65dims_lst))
    print(all65dims_lst)
    # load data
    human_data = (pd.read_excel(BBSR_file, usecols=['Word'] + all65dims_lst,
                                na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                  .rename(columns={'Word': 'Concept'})
                  .fillna(0)
                  .drop_duplicates()
                  .query("Concept != 'used'")
                  .set_index('Concept'))
    # print(human_data)
    LLM_data = (pd.read_excel(analysis_data_folder + LLM_MEAN_file, usecols=['Concept'] + all65dims_lst,
                              na_values=['', ' ', 'NA', 'n/a', '#N/A', '#NA', 'NaN', 'na'])
                .fillna(0)
                .drop_duplicates()
                .query("Concept != 'used'")
                .set_index('Concept'))



    # concep-level corr, kl, mse
    for domain in domain_dimlst_dic.keys():
        focus_dims = domain_dimlst_dic[domain]
        corr_kl_mse_data = calculate_concept_correlations_kl_mse(human_data, LLM_data, focus_dims)


        export_to_excel(corr_kl_mse_data, analysis_result_folder +'Domain_'+domain+'_corr_kl_mse.xlsx',
                        columns_lst = ['Concept', domain+'_corr', domain+"_mse", domain+"_kl"])


    alldims_corr_kl_mse_data = calculate_concept_correlations_kl_mse(human_data, LLM_data, all65dims_lst)


    export_to_excel(alldims_corr_kl_mse_data, analysis_result_folder +'ALLDomain_corr_kl_mse.xlsx',
                        columns_lst = ['Concept', 'corr', "mse", "kl"])



    plot_LLM_human_radar(human_data, LLM_data, all65dims_lst, LLM_name)




if __name__ == "__main__":
    BBSR_analysis("gpt-3.5-turbo", "BBSR_gpt-3.5-turbo[MEAN]TTW.xlsx")
    BBSR_analysis("gpt4omini", "BBSR_gpt-4o-mini[MEAN]TTW.xlsx")

    BBSR_analysis("Llama-3.1-8B-Instruct", "BBSR_Llama-3.1-8B-Instruct[MEAN]TTW.xlsx")
    BBSR_analysis("Llama-3.1-70B-Instruct", "BBSR_Llama-3.1-70B-Instruct[MEAN]TTW.xlsx")
    BBSR_analysis("Llama-3.1-405B-Instruct", "BBSR_Llama-3.1-405B-Instruct[MEAN]TTW.xlsx")

    BBSR_analysis("Llama-3-8B-Instruct", "BBSR_Llama-3-8B-Instruct[MEAN]TTW.xlsx")
    BBSR_analysis("Llama-3-70B-Instruct", "BBSR_Llama-3-70B-Instruct[MEAN]TTW.xlsx")


    BBSR_analysis("Qwen2-7B-Instruct", "BBSR_Qwen2-7B-Instruct[MEAN]TTW.xlsx")
    BBSR_analysis("Qwen2-72B-Instruct", "BBSR_Qwen2-72B-Instruct[MEAN]TTW.xlsx")


